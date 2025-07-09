from model import infer
import pandas as pd
import cv2
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt

# --- Configuration ---
CKPT = "iter_3000.pth"
CONFIG = "km_swin_t_config.py"
DEVICE = "cuda"

# --- Initialize the OCR reader once at the start ---
print("Initializing EasyOCR reader... (This may take a moment on first run)")
ocr_reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR reader initialized.")


def run_line_finder(img):
    """Runs the line segmentation model to get pixel coordinates of lines."""
    print("--- Running Line Segmentation Model ---")
    line_dataseries, inst_masks = infer.get_dataseries(img, to_clean=False, return_masks=True)
    print(f"Found {len(line_dataseries)} potential data lines.")
    return line_dataseries


def run_plot_detector(img_path):
    """Runs the YOLO object detection model to find plot components."""
    print("\n--- Running YOLO Plot Info Detector ---")
    model = YOLO("../model/yoloPlotInfoDetector.pt")
    results = model(img_path)
    print(f"Detected {len(results[0].boxes)} objects.")
    return results[0]


def get_text_for_bbox(image_np, bbox, reader, debug=False, rotate=False):
    """
    Crops an image to the given bounding box, processes it, and runs OCR using EasyOCR.
    If debug is True, it will display a detailed 3-panel view of the process.
    If rotate is True, it will rotate the image 90 degrees counter-clockwise.
    """
    # 1. Crop the image to the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    original_crop = image_np[y1:y2, x1:x2]

    if original_crop.size == 0:
        return ""

    # --- NEW: Rotate the image if requested (for vertical text) ---
    if rotate:
        original_crop = cv2.rotate(original_crop, cv2.ROTATE_90_CLOCKWISE)

    # 2. Apply pre-processing steps
    scale_factor = 2
    resized_crop = cv2.resize(original_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    blurred_crop = cv2.blur(resized_crop, (5, 5))

    gray_image = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    margin = 2
    # Use BORDER_CONSTANT with value=0 for a black border
    thresh_image_with_border = cv2.copyMakeBorder(
        thresh_image,
        top=margin,
        bottom=margin,
        left=margin,
        right=margin,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # 3. Configure and run EasyOCR
    allowlist = '0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz() '
    result = reader.readtext(
        thresh_image_with_border,
        detail=0,
        paragraph=True,
        allowlist=allowlist
    )
    result_text = " ".join(result) if result else ""

    # 4. Show the detailed debug visualization if enabled
    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        fig.suptitle(f"OCR Result: '{result_text}'", fontsize=16, y=1.02)

        # Panel 1: The original, unaltered crop (will be rotated if applicable)
        ax1.imshow(cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB))
        ax1.set_title("1. Original Crop (Rotated)" if rotate else "1. Original Crop")
        ax1.axis('off')

        # Panel 2: The resized and blurred intermediate image
        ax2.imshow(cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2RGB))
        ax2.set_title("2. Resized & Blurred")
        ax2.axis('off')

        # Panel 3: The final black & white image sent to OCR
        ax3.imshow(thresh_image_with_border, cmap='gray')
        ax3.set_title("3. Final Input to OCR")
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    return result_text


def create_axis_scaler(ticks, axis_type='x'):
    """
    Creates a linear scaler function from a list of tick information.
    It automatically inverts the scale for the 'y' axis.
    """
    if len(ticks) < 2:
        return None

    # For both axes, we find the min/max of the pixel and data values.
    pixel_coords = [t['pixel'] for t in ticks]
    data_values = [t['value'] for t in ticks]

    pixel_min, pixel_max = min(pixel_coords), max(pixel_coords)
    value_min, value_max = min(data_values), max(data_values)

    if pixel_max == pixel_min:
        return None

    def scaler(pixel_coord):
        # Calculate the position of the pixel within the pixel range (0.0 to 1.0)
        pixel_ratio = (pixel_coord - pixel_min) / (pixel_max - pixel_min)

        if axis_type == 'y':
            # --- FIX: Invert the mapping for the y-axis ---
            # A low pixel_ratio (top of image) should map to a high data value.
            # A high pixel_ratio (bottom of image) should map to a low data value.
            data_value = value_max - pixel_ratio * (value_max - value_min)
        else:
            # For the x-axis, the mapping is direct.
            data_value = value_min + pixel_ratio * (value_max - value_min)
        return data_value

    return scaler


def extract_data_from_plot(line_dataseries, yolo_results, full_image, reader, debug_ocr=False):
    """
    Translates pixel coordinates from lines into actual data values using YOLO box info.
    """
    print("\n--- Translating Pixels to Data Values ---")
    boxes = yolo_results.boxes
    names = yolo_results.names

    x_axis_ticks = []
    y_axis_ticks = []
    # --- NEW: Initialize axis titles with default values ---
    x_axis_title = "X-Axis"
    y_axis_title = "Y-Axis"

    plot_area_box = None
    for box in boxes:
        if names[int(box.cls)] == 'plot_area':
            plot_area_box = box.xyxy[0]
            break

    if plot_area_box is None:
        print("Warning: No 'plot_area' detected. Cannot reliably separate axes.")
        h, w, _ = full_image.shape
        plot_area_box = [0, 0, w, h]

    plot_x_center = (plot_area_box[0] + plot_area_box[2]) / 2
    plot_y_center = (plot_area_box[1] + plot_area_box[3]) / 2

    print("Extracting text from tick labels and titles using OCR...")
    for box in boxes:
        class_name = names[int(box.cls)]

        if class_name == 'tick_label':
            tick_text = get_text_for_bbox(full_image, box.xyxy[0], reader, debug=debug_ocr)
            try:
                tick_value = float(tick_text)
                box_coords = box.xywh[0]
                box_x, box_y = box_coords[0], box_coords[1]

                if box_x < plot_x_center and abs(box_y - plot_y_center) < plot_y_center * 1.5:
                    y_axis_ticks.append({'value': tick_value, 'pixel': int(box_y)})
                elif box_y > plot_y_center:
                    x_axis_ticks.append({'value': tick_value, 'pixel': int(box_x)})

            except (ValueError, IndexError):
                print(f"  - Could not parse tick OCR result '{tick_text}' to a number. Skipping.")
                continue
        # --- NEW: Handle axis_title class ---
        elif class_name == 'axis_title':
            box_coords = box.xywh[0]
            box_x, box_y = box_coords[0], box_coords[1]

            # Y-axis titles are typically to the left of the plot
            if box_x < plot_x_center:
                # --- NEW: Call OCR with rotation enabled for the Y-axis title ---
                title_text = get_text_for_bbox(full_image, box.xyxy[0], reader, debug=debug_ocr, rotate=True)
                if title_text:
                    y_axis_title = title_text
            # X-axis titles are typically below the plot
            elif box_y > plot_y_center:
                title_text = get_text_for_bbox(full_image, box.xyxy[0], reader, debug=debug_ocr, rotate=False)
                if title_text:
                    x_axis_title = title_text

    print(f"Found {len(y_axis_ticks)} Y-axis ticks and {len(x_axis_ticks)} X-axis ticks.")
    print(f"Found X-axis title: '{x_axis_title}'")
    print(f"Found Y-axis title: '{y_axis_title}'")

    y_scaler = create_axis_scaler(y_axis_ticks, axis_type='y')
    x_scaler = create_axis_scaler(x_axis_ticks, axis_type='x')

    if not y_scaler or not x_scaler:
        print("Error: Could not create axis scalers. Not enough valid tick labels were read.")
        return None

    all_series_data = []
    for i, line in enumerate(line_dataseries):
        series_data = []
        for point in line:
            px, py = point['x'], point['y']
            data_x = x_scaler(px)
            data_y = y_scaler(py)
            series_data.append({'x': round(data_x, 2), 'y': round(data_y, 2)})

        all_series_data.append({
            'series_name': f'series_{i + 1}',
            'data_points': series_data
        })

    # --- NEW: Return a dictionary containing all extracted information ---
    return {
        'series_data': all_series_data,
        'x_axis_title': x_axis_title,
        'y_axis_title': y_axis_title
    }


if __name__ == '__main__':
    infer.load_model(CONFIG, CKPT, DEVICE)

    img_path = "../demo/Graph.png"
    img = cv2.imread(img_path)

    # --- STAGE 1: Find Lines (in pixels) ---
    line_dataseries = run_line_finder(img)

    # --- STAGE 2: Find Plot Components ---
    yolo_results = run_plot_detector(img_path)

    # --- STAGE 3: Combine and Translate ---
    extraction_result = extract_data_from_plot(line_dataseries, yolo_results, img, ocr_reader, debug_ocr=False)

    # --- STAGE 4: Display Results ---
    if extraction_result:
        series_data = extraction_result['series_data']
        print("\n---  Final Extracted Table Data ---")
        for series in series_data:
            print(f"\n--- {series['series_name']} ---")
            df = pd.DataFrame(series['data_points'])
            print(df)
    else:
        print("\n---  Could not extract data from the plot. ---")

    # Gather all the data and recreate the plot from the extracted data
    if extraction_result:
        plt.figure(figsize=(10, 6))
        for series in extraction_result['series_data']:
            df = pd.DataFrame(series['data_points'])
            plt.plot(df['x'], df['y'], marker='o', label=series['series_name'])

        # --- NEW: Use the extracted titles for the plot labels ---
        plt.xlabel(extraction_result['x_axis_title'])
        plt.ylabel(extraction_result['y_axis_title'])
        plt.title("Recreated Plot from Extracted Data")
        plt.legend()
        plt.grid(True)
        plt.show()
