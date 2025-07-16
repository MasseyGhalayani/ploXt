# chart_extractor.py

import argparse
import io
import itertools
from pathlib import Path

import cv2
import easyocr
# Set Matplotlib backend to a non-interactive one for thread safety
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# Assuming 'infer' is a local module you have
from model import infer


class ChartExtractor:
    """
    A class to encapsulate the entire chart data extraction pipeline.
    It loads all necessary models upon initialization and provides methods
    to process an image and recalculate data from corrected inputs.
    """

    def __init__(self, yolo_model_path=None, line_model_config=None,
                 line_model_ckpt=None, device="cuda"):
        """
        Initializes the ChartExtractor by loading all required models.
        """
        print("--- Initializing Chart Extractor ---")
        script_dir = Path(__file__).parent.resolve()

        # Set default paths if they are not provided
        if yolo_model_path is None:
            yolo_model_path = script_dir / "yoloPlotInfoDetector.pt"
        if line_model_config is None:
            line_model_config = script_dir / "km_swin_t_config.py"
        if line_model_ckpt is None:
            line_model_ckpt = script_dir / "iter_3000.pth"

        # 1. Load Line Segmentation Model
        print(f"Loading line segmentation model from config: {line_model_config}")
        infer.load_model(str(line_model_config), str(line_model_ckpt), device)

        # 2. Load YOLO Model
        print(f"Loading YOLO model from: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)

        # 3. Load OCR Reader
        print("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        print("--- Chart Extractor Initialized Successfully ---")

    def _run_line_finder(self, img):
        """
        Runs the line segmentation model to get raw pixel coordinates of lines.
        --- MODIFIED: Now returns the instance masks and raw keypoints (uninterpolated). ---
        """
        print("--- Running Line Segmentation Model ---")
        # Get raw keypoints by disabling interpolation
        raw_keypoints, inst_masks = infer.get_dataseries(img, to_clean=False, return_masks=True, interpolation_type=None)
        print(f"Found {len(raw_keypoints)} potential data lines (raw keypoints).")
        return raw_keypoints, inst_masks

    def _run_plot_detector(self, img_or_path):
        """Runs the YOLO object detection model to find plot components."""
        print("\n--- Running YOLO Plot Info Detector ---")
        results = self.yolo_model(img_or_path)
        print(f"Detected {len(results[0].boxes)} objects.")
        return results[0]

    def _get_text_for_bbox(self, image_np, bbox, ocr_params=None, allowlist=None, debug=False, rotate=False):
        """
        Crops an image, processes it, and runs OCR using the trusted logic.
        Returns the text, the original un-processed crop, and the final processed image sent to OCR.
        """
        if ocr_params is None:
            ocr_params = {}

        # --- NEW: Define default general allowlist if none is provided ---
        if allowlist is None:
            allowlist = '0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() '

        x1, y1, x2, y2 = map(int, bbox)
        original_crop = image_np[y1:y2, x1:x2]

        if original_crop.size == 0:
            return "", None, None

        if rotate:
            original_crop = cv2.rotate(original_crop, cv2.ROTATE_90_CLOCKWISE)

        # Pre-processing steps
        scale_factor = ocr_params.get('scale_factor', 2)
        blur_ksize = ocr_params.get('blur_ksize', 5)
        margin = ocr_params.get('margin', 2)

        # Ensure blur kernel size is odd
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        resized_crop = cv2.resize(original_crop, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_LINEAR)
        blurred_crop = cv2.blur(resized_crop, (blur_ksize, blur_ksize))
        gray_image = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        thresh_image_with_border = cv2.copyMakeBorder(
            thresh_image, top=margin, bottom=margin, left=margin, right=margin,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # OCR Execution
        result = self.ocr_reader.readtext(
            thresh_image_with_border, detail=0, paragraph=True, allowlist=allowlist
        )
        result_text = " ".join(result) if result else ""

        if debug:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            fig.suptitle(f"OCR Result: '{result_text}'", fontsize=16, y=1.02)
            ax1.imshow(cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB))
            ax1.set_title("1. Original Crop (Rotated)" if rotate else "1. Original Crop")
            ax1.axis('off')
            ax2.imshow(cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2RGB))
            ax2.set_title("2. Resized & Blurred")
            ax2.axis('off')
            ax3.imshow(thresh_image_with_border, cmap='gray')
            ax3.set_title("3. Final Input to OCR")
            ax3.axis('off')
            plt.tight_layout()
            plt.show()

        return result_text, original_crop, thresh_image_with_border

    def ocr_on_region(self, image_np, bbox, ocr_params=None, rotate=False):
        """
        Public method to perform OCR on a specified bounding box of an image with custom parameters.
        Returns the new text and the new processed image.
        --- MODIFIED: Uses the general-purpose allowlist by default. ---
        """
        print(f"--- Performing manual OCR on region: {bbox} with params: {ocr_params} ---")
        # Manual selection is for titles, so use the general allowlist.
        general_allowlist = '0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() '
        text, _, processed_img = self._get_text_for_bbox(
            image_np, bbox, ocr_params=ocr_params, allowlist=general_allowlist, rotate=rotate)
        return text, processed_img

    def rerun_all_ocr(self, full_image, yolo_results, ocr_params):
        """
        Re-runs only the OCR extraction step on all detected items with new parameters.
        """
        print(f"--- Re-running all OCR with new global parameters: {ocr_params} ---")
        updated_ocr_data = self._extract_ocr_data(yolo_results, full_image, ocr_params=ocr_params)
        return updated_ocr_data

    def _find_best_ocr_params(self, yolo_results, full_image):
        """
        Automatically tests different OCR pre-processing parameters to find the combination
        that yields the most numerous and consistent tick labels.
        """
        print("\n--- Finding optimal OCR parameters by auto-tuning ---")

        param_grid = {
            'scale_factor': [1.5, 2.0, 2.5],
            'blur_ksize': [1, 3, 5],
            'margin': [2, 4]
        }

        best_score = -1
        best_params = {}

        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        def calculate_axis_score(values):
            num_values = len(values)
            if num_values < 2:
                return float(num_values)

            sorted_values = np.array(sorted(values))
            diffs = np.diff(sorted_values)

            # Check for uniform spacing, which is a strong signal of correctness
            if len(diffs) > 0:
                mean_diff = np.mean(diffs)
                if mean_diff > 1e-9:
                    # Coefficient of variation of the differences
                    coeff_var = np.std(diffs) / mean_diff
                    # Score is higher for more ticks and lower variation.
                    # The penalty for variation is significant.
                    return num_values / (1 + coeff_var)
            return float(num_values) # Fallback score

        for params in param_combinations:
            temp_ocr_data = self._extract_ocr_data(yolo_results, full_image, ocr_params=params)
            x_ticks = [t['value'] for t in temp_ocr_data.get('ticks', []) if t.get('axis') == 'x']
            y_ticks = [t['value'] for t in temp_ocr_data.get('ticks', []) if t.get('axis') == 'y']
            total_score = calculate_axis_score(x_ticks) + calculate_axis_score(y_ticks)
            if total_score > best_score:
                best_score = total_score
                best_params = params
        print(f"--- Best OCR parameters found: {best_params} with score {best_score:.2f} ---")
        return best_params

    @staticmethod
    def _create_axis_scaler(ticks, axis_type='x'):
        """
        Creates a linear scaler function from a list of tick information using linear regression.
        This is more robust to OCR outliers than a simple min/max mapping.
        """
        if len(ticks) < 2:
            return None

        # Extract pixel and data value pairs for regression
        pixel_coords = np.array([t['pixel'] for t in ticks])
        data_values = np.array([t['value'] for t in ticks])

        # Use linear regression (polyfit with degree 1) to find the best-fit line
        # that maps pixel coordinates to data values.
        # The model is: data_value = m * pixel_coord + c
        # np.polyfit returns the coefficients [m, c]
        try:
            m, c = np.polyfit(pixel_coords, data_values, 1)
        except (np.linalg.LinAlgError, ValueError):
            # This can happen if the input data is degenerate (e.g., all points on a vertical line).
            print(f"Warning: Linear regression failed for {axis_type}-axis. Cannot create scaler.")
            return None

        # The scaler function simply applies the linear equation found by the regression.
        # This works for both X and Y axes, as the slope 'm' will be positive for X
        # and negative for Y (since pixel Y increases downwards).
        def scaler(pixel_coord):
            return m * pixel_coord + c

        return scaler

    def _extract_ocr_data(self, yolo_results, full_image, ocr_params=None, debug_ocr=False):
        """
        Extracts OCR data and structures it for the GUI.
        Accepts ocr_params to allow for dynamic adjustments.
        --- MODIFIED: Uses specialized allowlists for different text types. ---
        """
        print("\n--- Extracting OCR Data ---")
        if ocr_params is None:
            ocr_params = {}
        boxes = yolo_results.boxes
        names = yolo_results.names

        # --- NEW: Define specialized allowlists ---
        allowlist_general = '0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() '
        allowlist_digits = '0123456789.-'

        ocr_data = {
            "plot_title": {"text": "", "bbox": None},
            "x_axis_title": {"text": "X-Axis", "bbox": None},
            "y_axis_title": {"text": "Y-Axis", "bbox": None},
            "ticks": [],
            "ocr_debug_info": []  # New key for debug data
        }

        plot_area_box = next((box.xyxy[0] for box in boxes if names[int(box.cls)] == 'plot_area'), None)
        if plot_area_box is None:
            print("Warning: No 'plot_area' detected.")
            h, w, _ = full_image.shape
            plot_area_box = [0, 0, w, h]

        plot_x_center = (plot_area_box[0] + plot_area_box[2]) / 2
        plot_y_center = (plot_area_box[1] + plot_area_box[3]) / 2

        for box in boxes:
            class_name = names[int(box.cls)]
            bbox = box.xyxy[0].tolist()
            box_coords_xywh = box.xywh[0]
            box_x, box_y = box_coords_xywh[0], box_coords_xywh[1]

            # Determine if rotation is needed for axis titles before calling OCR
            is_y_axis_title = class_name == 'axis_title' and box_x < plot_x_center

            # --- NEW: Select allowlist based on class name ---
            current_allowlist = allowlist_digits if class_name == 'tick_label' else allowlist_general

            # --- MODIFIED: Pass the selected allowlist to the OCR function ---
            text, original_crop, processed_img = self._get_text_for_bbox(
                full_image, bbox, ocr_params=ocr_params, allowlist=current_allowlist,
                debug=debug_ocr, rotate=is_y_axis_title
            )

            if class_name == 'tick_label':
                if original_crop is None:
                    continue
                try:
                    value = float(text)
                    tick_data = {
                        "text": text, "value": value,
                        # --- FIX: Use original_crop which is the correct variable name now ---
                        "crop_image": original_crop,
                        "pixel_x": int(box_x), "pixel_y": int(box_y), "bbox": bbox,
                        "axis": "unknown"
                    }
                    # --- FIX: Use a more robust geometric condition for axis classification. ---
                    # The old method was unreliable for plots not centered in the image.
                    # This new logic checks if a tick is on the left half AND vertically
                    # within the plot area's boundaries to be a Y-tick. Otherwise, if it's
                    # below the plot's center, it's an X-tick. This is much more robust.
                    plot_y1 = plot_area_box[1]
                    plot_y2 = plot_area_box[3]

                    if box_x < plot_x_center and (plot_y1 < box_y < plot_y2):
                        tick_data['axis'] = 'y'
                    elif box_y > plot_y_center:
                        tick_data['axis'] = 'x'

                    if tick_data['axis'] != 'unknown':
                        ocr_data["ticks"].append(tick_data)
                except (ValueError, IndexError):
                    print(f"  - Could not parse tick OCR result '{text}' to a number. Skipping.")

            elif class_name == 'axis_title':
                # --- NEW: For titles, compare auto-tuned OCR with standard OCR and pick the best ---
                # Run with auto-tuned parameters first (result is in 'text' variable)
                text_tuned = text

                # Run with a set of standard, general-purpose parameters
                standard_params = {'scale_factor': 2.0, 'blur_ksize': 3, 'margin': 4}
                text_standard, _, _ = self._get_text_for_bbox(
                    full_image, bbox, ocr_params=standard_params, allowlist=current_allowlist,
                    debug=False, rotate=is_y_axis_title
                )

                # Choose the result with more characters, as it's likely more complete.
                final_text = text_standard if len(text_standard) > len(text_tuned) else text_tuned

                if final_text:
                    if is_y_axis_title:
                        ocr_data["y_axis_title"] = {"text": final_text, "bbox": bbox}
                    else:
                        ocr_data["x_axis_title"] = {"text": final_text, "bbox": bbox}

            # --- NEW: Store debug info for all OCR'd items ---
            if original_crop is not None:
                ocr_data["ocr_debug_info"].append({
                    'label': f'{class_name}: {text[:15]}...',
                    'original_crop': original_crop,
                    'processed_img': processed_img,
                    'text': text,
                    'bbox': bbox,
                    'rotate': is_y_axis_title
                })
        return ocr_data

    @staticmethod
    def _recreate_plot_image(extraction_result):
        """Generates a plot from the extracted data and returns it as a NumPy image array."""
        if not extraction_result or not extraction_result.get('series_data'):
            return None

        plt.figure(figsize=(8, 5))
        # --- FIX: Get the colors from the result dictionary if available ---
        plot_colors = extraction_result.get('colors')

        for i, series in enumerate(extraction_result['series_data']):
            df = pd.DataFrame(series['data_points'])
            if not df.empty:
                # Use the provided color if available, otherwise let matplotlib decide
                # --- FIX: Use the 'original_index' to ensure color consistency after deletion ---
                original_index = series.get('original_index', i)
                color = plot_colors[original_index % len(plot_colors)] if plot_colors else None
                plt.plot(df['x'], df['y'], linestyle='-', label=series.get('series_name', 'series'), color=color)

        plt.xlabel(extraction_result['x_axis_title'])
        plt.ylabel(extraction_result['y_axis_title'])
        plt.title(extraction_result.get('plot_title', 'Recreated Plot from Extracted Data'))
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_img_pil = Image.open(buf)
        return cv2.cvtColor(np.array(plot_img_pil), cv2.COLOR_RGBA2BGRA)

    def process_image(self, image_or_path, debug_ocr=False, ocr_params=None):
        """
        The main public method to run the full extraction pipeline on a single image.
        --- MODIFIED: Now returns intermediate results even if final calculation fails. ---
        --- MODIFIED: Accepts optional ocr_params to skip auto-tuning. ---
        """
        try:
            if isinstance(image_or_path, (str, Path)):
                img_path = Path(image_or_path)
                if not img_path.is_file():
                    raise FileNotFoundError(f"Input image not found at: {img_path}")
                img = cv2.imread(str(img_path))
            else:
                img = image_or_path
                img_path = None

            raw_keypoints, inst_masks = self._run_line_finder(img)
            yolo_results = self._run_plot_detector(img_path if img_path else img)
            
            # --- MODIFIED: Use provided OCR params if available, otherwise find them ---
            if ocr_params:
                print("--- Using provided OCR parameters, skipping auto-tuning ---")
                best_ocr_params = ocr_params
            else:
                # Auto-tune OCR parameters before final extraction
                best_ocr_params = self._find_best_ocr_params(yolo_results, img)

            ocr_data = self._extract_ocr_data(yolo_results, img, ocr_params=best_ocr_params, debug_ocr=debug_ocr)
            ocr_data['best_params_found'] = best_ocr_params

            # --- FIX: The first call to recalculate also needs to provide indexed keypoints ---
            # This ensures the 'original_index' is present from the very first run.
            initial_lines_to_process = list(enumerate(raw_keypoints))

            # This is the part that can fail if OCR is poor
            calc_results = self.recalculate_from_corrected(
                initial_lines_to_process, ocr_data, interpolation_method='linear'
            )

            # Build the final result dictionary, ensuring intermediate data is always present
            final_results = {
                'raw_keypoints': raw_keypoints,
                'instance_masks': inst_masks,
                'ocr_data': ocr_data,
                'yolo_results': yolo_results,  # For re-running OCR
                'status': calc_results.get('status', 'error'),
                'message': calc_results.get('message', 'Calculation failed. Check OCR corrections.')
            }

            # Merge the successful calculation results if they exist
            if final_results['status'] == 'success':
                final_results.update(calc_results)

            return final_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': f'An unexpected error occurred: {e}'}

    def recalculate_from_corrected(self, lines_to_process, corrected_ocr_data, interpolation_method='linear'):
        """
        Takes raw keypoints, corrected OCR data, and an interpolation method,
        and re-runs the interpolation, scaling, and plotting steps.
        --- MODIFIED: Accepts (original_index, keypoints) tuples to preserve color mapping. ---
        """
        print(f"\n--- Recalculating with Corrected Data (Interpolation: {interpolation_method}) ---")

        interpolated_lines = []
        if interpolation_method and interpolation_method != 'none':
            for original_index, line_kps in lines_to_process:
                interpolated_line = infer.interpolate(line_kps, inter_type=interpolation_method)
                interpolated_lines.append((original_index, interpolated_line))
        else:
            interpolated_lines = lines_to_process

        x_axis_ticks = [{'value': t['value'], 'pixel': t['pixel_x']} for t in corrected_ocr_data['ticks'] if t.get('axis') == 'x']
        y_axis_ticks = [{'value': t['value'], 'pixel': t['pixel_y']} for t in corrected_ocr_data['ticks'] if t.get('axis') == 'y']

        y_scaler = self._create_axis_scaler(y_axis_ticks, axis_type='y')
        x_scaler = self._create_axis_scaler(x_axis_ticks, axis_type='x')

        if not y_scaler or not x_scaler:
            return {'status': 'error', 'message': 'Could not create axis scalers from corrected data.'}

        all_series_data = []
        for original_index, line in interpolated_lines:
            series_data = [{'x': x_scaler(p['x']), 'y': y_scaler(p['y'])} for p in line]
            all_series_data.append({
                'series_name': f'series_{original_index + 1}',
                'data_points': series_data,
                'original_index': original_index  # Preserve the original index
            })

        extraction_result = {'series_data': all_series_data, 'plot_title': corrected_ocr_data.get('plot_title', {}).get('text', 'Recreated Plot'), 'x_axis_title': corrected_ocr_data['x_axis_title']['text'], 'y_axis_title': corrected_ocr_data['y_axis_title']['text']}
        dataframes = [pd.DataFrame(s['data_points']) for s in extraction_result['series_data']]
        return {'status': 'success', 'message': 'Recalculation complete.', 'dataframes': dataframes, 'series_data': extraction_result['series_data'], 'plot_title': extraction_result['plot_title'], 'x_axis_title': extraction_result['x_axis_title'], 'y_axis_title': extraction_result['y_axis_title']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract data from a line plot image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input plot image.")
    parser.add_argument("--debug", action="store_true", help="Enable OCR debug visualizations.")
    parser.add_argument("--no-plot", action="store_true", help="Do not display the recreated plot at the end.")
    args = parser.parse_args()

    extractor = ChartExtractor()
    result = extractor.process_image(args.image, debug_ocr=args.debug)

    if result['status'] == 'success':
        print("\n--- Final Extracted Table Data ---")
        for series in result['series_data']:
            print(f"\n--- {series['series_name']} ---")
            df = pd.DataFrame(series['data_points'])
            print(df.to_string())
            print("-" * 30)

        if not args.no_plot:
            plot_image = ChartExtractor._recreate_plot_image(result)
            if plot_image is not None:
                print("\nDisplaying recreated plot. Press any key to exit.")
                cv2.imshow("Recreated Plot", plot_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print(f"\n--- Error during extraction ---")
        print(result['message'])