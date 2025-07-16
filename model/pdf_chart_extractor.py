# model/pdf_chart_extractor.py

from pathlib import Path
import cv2
import numpy as np
import pymupdf  # PyMuPDF
from ultralytics import YOLO

class PdfChartExtractor:
    """
    A class to find and extract chart images from PDF files.
    """
    def __init__(self, yolo_model_path=None, device="cuda"):
        """
        Initializes the extractor by loading the YOLO model.
        """
        print("--- Initializing PDF Chart Extractor ---")
        script_dir = Path(__file__).parent.resolve()
        if yolo_model_path is None:
            yolo_model_path = script_dir / "yoloPlotInfoDetector.pt"

        print(f"Loading YOLO model from: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.names = self.yolo_model.names
        print("--- PDF Chart Extractor Initialized Successfully ---")

    def _group_detections_by_plot(self, boxes):
        """
        Groups YOLO detections around each 'plot_area'.
        --- MODIFIED: Now only uses 'tick_label's in addition to the 'plot_area'
        to define the chart's boundary, ignoring axis titles for a more
        precise and reliable crop.
        """
        plot_areas = [box for box in boxes if self.names[int(box.cls)] == 'plot_area']
        # --- CHANGE: Limit components for boundary definition to just tick labels ---
        other_components = [box for box in boxes if self.names[int(box.cls)] == 'tick_label']

        if not plot_areas:
            return []

        chart_groups = []
        for plot_area_box in plot_areas:
            pa_x1, pa_y1, pa_x2, pa_y2 = plot_area_box.xyxy[0].tolist()

            # Define a generous search area around the plot_area. This is useful for
            # associating the correct ticks with the correct plot area, especially
            # on pages that contain multiple charts.
            search_x_margin = (pa_x2 - pa_x1) * 0.5
            search_y_margin = (pa_y2 - pa_y1) * 0.5
            search_x1 = pa_x1 - search_x_margin
            search_y1 = pa_y1 - search_y_margin
            search_x2 = pa_x2 + search_x_margin
            search_y2 = pa_y2 + search_y_margin * 2.0  # Extra margin below for x-axis ticks

            current_chart_components = [plot_area_box]

            for component_box in other_components:
                c_x, c_y = component_box.xywh[0][:2].tolist()
                # Check if the component's center is within the search area
                if search_x1 < c_x < search_x2 and search_y1 < c_y < search_y2:
                    current_chart_components.append(component_box)

            chart_groups.append(current_chart_components)

        return chart_groups

    def _get_encompassing_bbox(self, component_boxes, margin_px=20):
        """Calculates a single bounding box that encloses all component boxes."""
        if not component_boxes:
            return None

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        for box in component_boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            min_x, min_y, max_x, max_y = min(min_x, x1), min(min_y, y1), max(max_x, x2), max(max_y, y2)
            
        # Add margin and ensure coordinates are within image bounds
        bbox = [max(0, min_x - margin_px), max(0, min_y - margin_px), max_x + margin_px, max_y + margin_px]
        return list(map(int, bbox))

    def extract_charts_from_pdf(self, pdf_path):
        """
        Main public method to run a two-pass chart extraction from a PDF.
        Pass 1: Identifies candidate regions on the full page.
        Pass 2: Validates and refines the crop for each candidate.
        Returns a list of final, validated chart images as NumPy arrays.
        """
        print(f"--- Processing PDF: {pdf_path} ---")
        extracted_charts = []
        try:
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    print(f"  - Processing page {page_num + 1}/{len(doc)}")
                    
                    pix = page.get_pixmap(dpi=200)
                    page_img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, -1)
                    page_img_np = cv2.cvtColor(page_img_np, cv2.COLOR_BGRA2BGR if pix.alpha else cv2.COLOR_RGB2BGR)
                    page_h, page_w = page_img_np.shape[:2]

                    # --- PASS 1: Find candidate regions with a generous crop ---
                    results_pass1 = self.yolo_model(page_img_np, verbose=False)[0]
                    chart_candidate_groups = self._group_detections_by_plot(results_pass1.boxes)
                    print(f"    Pass 1: Found {len(chart_candidate_groups)} potential chart regions.")

                    for i, group in enumerate(chart_candidate_groups):
                        # Get a generous initial crop box
                        initial_bbox = self._get_encompassing_bbox(group, margin_px=40)
                        if not initial_bbox: continue
                        
                        ix1, iy1, ix2, iy2 = initial_bbox
                        if not (ix2 > ix1 and iy2 > iy1): continue
                        
                        initial_crop_img = page_img_np[iy1:iy2, ix1:ix2]
                        if initial_crop_img.size == 0: continue

                        # --- PASS 2: Validate the crop and refine the bounding box ---
                        results_pass2 = self.yolo_model(initial_crop_img, verbose=False)[0]
                        
                        # Validate: must have at least one plot_area and two tick_labels
                        refined_plot_areas = [b for b in results_pass2.boxes if self.names[int(b.cls)] == 'plot_area']
                        refined_ticks = [b for b in results_pass2.boxes if self.names[int(b.cls)] == 'tick_label']

                        if not refined_plot_areas or len(refined_ticks) < 2:
                            print(f"      - Candidate {i+1} failed validation (plot area or ticks missing). Skipping.")
                            continue
                        
                        print(f"      - Candidate {i+1} passed validation. Refining crop...")
                        
                        # Get the tight bounding box of all components within the crop
                        all_refined_components = refined_plot_areas + refined_ticks
                        tight_bbox_in_crop = self._get_encompassing_bbox(all_refined_components, margin_px=15)
                        
                        # Find the center of the largest plot area to anchor the final crop
                        main_refined_pa_box = max(refined_plot_areas, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                        pa_x1, pa_y1, pa_x2, pa_y2 = main_refined_pa_box.xyxy[0].tolist()
                        pa_center_x = (pa_x1 + pa_x2) / 2
                        pa_center_y = (pa_y1 + pa_y2) / 2
                        
                        # Calculate final crop dimensions based on all components, but centered on the plot area
                        tight_w = tight_bbox_in_crop[2] - tight_bbox_in_crop[0]
                        tight_h = tight_bbox_in_crop[3] - tight_bbox_in_crop[1]

                        # Ensure the final crop is centered on the plot area
                        fx1 = int(pa_center_x - tight_w / 2)
                        fy1 = int(pa_center_y - tight_h / 2)
                        fx2 = int(pa_center_x + tight_w / 2)
                        fy2 = int(pa_center_y + tight_h / 2)

                        # Clip coordinates to the bounds of the initial crop
                        crop_h, crop_w = initial_crop_img.shape[:2]
                        final_bbox = [max(0, fx1), max(0, fy1), min(crop_w, fx2), min(crop_h, fy2)]

                        if final_bbox[2] > final_bbox[0] and final_bbox[3] > final_bbox[1]:
                            final_chart_img = initial_crop_img[final_bbox[1]:final_bbox[3], final_bbox[0]:final_bbox[2]]
                            extracted_charts.append(final_chart_img)
                            print(f"        ...Extracted chart {len(extracted_charts)} successfully.")
                            
        except Exception as e:
            import traceback
            print(f"Error processing PDF: {e}\n{traceback.format_exc()}")

        print(f"--- Finished PDF processing. Extracted {len(extracted_charts)} charts in total. ---")
        return extracted_charts