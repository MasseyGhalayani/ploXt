# chart_extractor.py

import argparse
import io
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

    def _get_text_for_bbox(self, image_np, bbox, debug=False, rotate=False):
        """
        Crops an image, processes it, and runs OCR using the trusted logic.
        Returns the cropped image along with the text.
        """
        x1, y1, x2, y2 = map(int, bbox)
        original_crop = image_np[y1:y2, x1:x2]

        if original_crop.size == 0:
            return "", None

        if rotate:
            original_crop = cv2.rotate(original_crop, cv2.ROTATE_90_CLOCKWISE)

        # Pre-processing steps
        scale_factor = 2
        resized_crop = cv2.resize(original_crop, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_LINEAR)
        blurred_crop = cv2.blur(resized_crop, (5, 5))
        gray_image = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        margin = 2
        thresh_image_with_border = cv2.copyMakeBorder(
            thresh_image, top=margin, bottom=margin, left=margin, right=margin,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # OCR Execution
        allowlist = '0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() '
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

        return result_text, original_crop

    def ocr_on_region(self, image_np, bbox):
        """Public method to perform OCR on a specified bounding box of an image."""
        print(f"--- Performing manual OCR on region: {bbox} ---")
        text, _ = self._get_text_for_bbox(image_np, bbox, rotate=False)
        return text

    @staticmethod
    def _create_axis_scaler(ticks, axis_type='x'):
        """Creates a linear scaler function from a list of tick information."""
        if len(ticks) < 2:
            return None
        pixel_coords = [t['pixel'] for t in ticks]
        data_values = [t['value'] for t in ticks]
        pixel_min, pixel_max = min(pixel_coords), max(pixel_coords)
        value_min, value_max = min(data_values), max(data_values)

        if pixel_max == pixel_min:
            return None

        def scaler(pixel_coord):
            pixel_ratio = (pixel_coord - pixel_min) / (pixel_max - pixel_min)
            if axis_type == 'y':
                return value_max - pixel_ratio * (value_max - value_min)
            else:
                return value_min + pixel_ratio * (value_max - value_min)

        return scaler

    def _extract_ocr_data(self, yolo_results, full_image, debug_ocr=False):
        """
        Extracts OCR data and structures it for the GUI.
        """
        print("\n--- Extracting OCR Data ---")
        boxes = yolo_results.boxes
        names = yolo_results.names

        ocr_data = {
            "plot_title": {"text": "", "bbox": None},
            "x_axis_title": {"text": "X-Axis", "bbox": None},
            "y_axis_title": {"text": "Y-Axis", "bbox": None},
            "ticks": []
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

            if class_name == 'tick_label':
                text, crop_image = self._get_text_for_bbox(full_image, bbox, debug=debug_ocr)
                if crop_image is None:
                    continue
                try:
                    value = float(text)
                    tick_data = {
                        "text": text, "value": value,
                        "crop_image": crop_image,
                        "pixel_x": int(box_x), "pixel_y": int(box_y), "bbox": bbox,
                        "axis": "unknown"
                    }
                    if box_x < plot_x_center and abs(box_y - plot_y_center) < plot_y_center * 1.5:
                        tick_data['axis'] = 'y'
                    elif box_y > plot_y_center:
                        tick_data['axis'] = 'x'

                    if tick_data['axis'] != 'unknown':
                        ocr_data["ticks"].append(tick_data)
                except (ValueError, IndexError):
                    print(f"  - Could not parse tick OCR result '{text}' to a number. Skipping.")

            elif class_name == 'axis_title':
                is_y_axis = box_x < plot_x_center
                text, _ = self._get_text_for_bbox(full_image, bbox, debug=debug_ocr, rotate=is_y_axis)
                if text:
                    if is_y_axis:
                        ocr_data["y_axis_title"] = {"text": text, "bbox": bbox}
                    else:
                        ocr_data["x_axis_title"] = {"text": text, "bbox": bbox}
        return ocr_data

    def recalculate_from_corrected(self, raw_keypoints, corrected_ocr_data, interpolation_method='linear'):
        """
        Takes raw keypoints, corrected OCR data, and an interpolation method,
        and re-runs the interpolation, scaling, and plotting steps.
        """
        print(f"\n--- Recalculating with Corrected Data (Interpolation: {interpolation_method}) ---")

        # --- NEW: Interpolation Step ---
        interpolated_lines = []
        if interpolation_method and interpolation_method != 'none':
            for line_kps in raw_keypoints:
                # Note: infer.interpolate is in the infer module
                interpolated_lines.append(infer.interpolate(line_kps, inter_type=interpolation_method))
        else:  # 'none' or None
            interpolated_lines = raw_keypoints

        x_axis_ticks = [
            {'value': t['value'], 'pixel': t['pixel_x']}
            for t in corrected_ocr_data['ticks'] if t.get('axis') == 'x'
        ]
        y_axis_ticks = [
            {'value': t['value'], 'pixel': t['pixel_y']}
            for t in corrected_ocr_data['ticks'] if t.get('axis') == 'y'
        ]
 
        y_scaler = self._create_axis_scaler(y_axis_ticks, axis_type='y')
        x_scaler = self._create_axis_scaler(x_axis_ticks, axis_type='x')

        if not y_scaler or not x_scaler:
            return {'status': 'error', 'message': 'Could not create axis scalers from corrected data.'}

        all_series_data = []
        for i, line in enumerate(interpolated_lines):
            series_data = [{'x': round(x_scaler(p['x']), 2), 'y': round(y_scaler(p['y']), 2)} for p in line]
            all_series_data.append({'series_name': f'series_{i + 1}', 'data_points': series_data})

        extraction_result = {
            'series_data': all_series_data,
            'plot_title': corrected_ocr_data.get('plot_title', {}).get('text', 'Recreated Plot'),
            'x_axis_title': corrected_ocr_data['x_axis_title']['text'],
            'y_axis_title': corrected_ocr_data['y_axis_title']['text']
        }

        dataframes = [pd.DataFrame(s['data_points']) for s in extraction_result['series_data']]

        return {
            'status': 'success',
            'message': 'Recalculation complete.',
            'dataframes': dataframes,
            'series_data': extraction_result['series_data'],
            'plot_title': extraction_result['plot_title'],
            'x_axis_title': extraction_result['x_axis_title'],
            'y_axis_title': extraction_result['y_axis_title']
        }

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
                color = plot_colors[i % len(plot_colors)] if plot_colors else None
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

    def process_image(self, image_or_path, debug_ocr=False):
        """
        The main public method to run the full extraction pipeline on a single image.
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
 
            # --- MODIFIED: Gets raw keypoints and instance masks ---
            raw_keypoints, inst_masks = self._run_line_finder(img)
            yolo_results = self._run_plot_detector(img_path if img_path else img)
            ocr_data = self._extract_ocr_data(yolo_results, img, debug_ocr)
            # --- MODIFIED: Initial calculation with default interpolation ---
            final_results = self.recalculate_from_corrected(raw_keypoints, ocr_data, interpolation_method='linear')

            if final_results['status'] == 'success':
                # --- NEW: Store raw keypoints for re-calculation ---
                final_results['raw_keypoints'] = raw_keypoints
                final_results['ocr_data'] = ocr_data
                # --- NEW: Add masks to the final result ---
                final_results['instance_masks'] = inst_masks

            return final_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': f'An unexpected error occurred: {e}'}


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