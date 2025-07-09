# main_gui.py
import pandas as pd
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QLineEdit, QFormLayout, QComboBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QMessageBox,
                             QGroupBox, QScrollArea, QAction, QToolBar, QActionGroup)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from model.chart_extractor import ChartExtractor
from .interactive_image_viewer import InteractiveImageViewer
from .gui_widgets import SeriesEditorWidget

# Try to import scipy for .mat export, handle if not found
try:
    from scipy.io import savemat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Worker(QThread):
    """Worker thread for running heavy model inference."""
    finished = pyqtSignal(dict)

    def __init__(self, extractor, image_data):
        super().__init__()
        self.extractor = extractor
        self.image_data = image_data

    def run(self):
        result = self.extractor.process_image(self.image_data)
        self.finished.emit(result)


class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Chart Data Extractor")
        self.setGeometry(100, 100, 1600, 900)
        self.extractor = None
        self.current_results = None
        self.series_widgets = []
        self.mask_overlay_pixmap = None # Cache for the overlay

        # --- NEW: Define a shared color palette for masks and plots ---
        self.SERIES_COLORS = [
            QColor(31, 119, 180, 120), QColor(255, 127, 14, 120), QColor(44, 160, 44, 120),
            QColor(214, 39, 40, 120), QColor(148, 103, 189, 120), QColor(140, 86, 75, 120)
        ] # R, G, B, Alpha (for overlay)

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Image Viewer) ---
        left_panel_group = QGroupBox("Chart Image & Correction")
        left_panel_layout = QVBoxLayout(left_panel_group)
        self.image_viewer = InteractiveImageViewer()
        left_panel_layout.addWidget(self.image_viewer)

        # --- Toolbar ---
        self.build_toolbar()

        # --- Right Panel (Controls and Results) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)

        # Top Controls Box
        controls_group = QGroupBox("Workflow")
        controls_layout = QHBoxLayout(controls_group)
        self.run_btn = QPushButton(" Run Inference on Edited Image")
        self.run_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.run_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.run_btn.setEnabled(False)
        controls_layout.addWidget(self.run_btn)

        # Results Tabs
        self.tabs = QTabWidget()
        self.build_correction_tab()
        self.build_results_tab()

        right_panel_layout.addWidget(controls_group)
        right_panel_layout.addWidget(self.tabs)

        main_layout.addWidget(left_panel_group, 2)
        main_layout.addWidget(right_panel_widget, 1)

        # --- Connections ---
        self.run_btn.clicked.connect(self.run_inference)
        self.apply_corrections_btn.clicked.connect(self.apply_all_corrections)
        self.image_viewer.regionSelected.connect(self.handle_manual_ocr)

        self.init_models()

    def build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        style = self.style()

        # File Actions
        self.load_action = QAction(style.standardIcon(style.SP_DirOpenIcon), "Open Image", self)
        self.load_action.triggered.connect(self.load_image)
        toolbar.addAction(self.load_action)

        self.export_csv_action = QAction(style.standardIcon(style.SP_DialogSaveButton), "Export as CSV", self)
        self.export_csv_action.triggered.connect(self.export_as_csv)
        self.export_csv_action.setEnabled(False)
        toolbar.addAction(self.export_csv_action)

        self.export_mat_action = QAction(style.standardIcon(style.SP_DialogSaveButton), "Export as .mat", self)
        self.export_mat_action.triggered.connect(self.export_as_mat)
        self.export_mat_action.setEnabled(SCIPY_AVAILABLE)
        toolbar.addAction(self.export_mat_action)
        if not SCIPY_AVAILABLE:
            self.export_mat_action.setToolTip("Scipy not installed. Pip install scipy.")

        toolbar.addSeparator()

        # Editing Tool Actions
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        self.brush_action = QAction(style.standardIcon(style.SP_CustomBase), "Brush Tool", self)
        self.brush_action.setCheckable(True)
        self.brush_action.setChecked(True)
        self.brush_action.triggered.connect(lambda: self.image_viewer.set_tool('brush'))
        toolbar.addAction(self.brush_action)
        tool_group.addAction(self.brush_action)

        self.fill_action = QAction(style.standardIcon(style.SP_CustomBase), "Select & Fill Tool", self)
        self.fill_action.setCheckable(True)
        self.fill_action.triggered.connect(lambda: self.image_viewer.set_tool('select_and_fill'))
        toolbar.addAction(self.fill_action)
        tool_group.addAction(self.fill_action)

        self.title_select_action = QAction(style.standardIcon(style.SP_CustomBase), "Select Plot Title", self)
        self.title_select_action.setCheckable(True)
        self.title_select_action.triggered.connect(lambda: self.image_viewer.set_tool('select_title'))
        toolbar.addAction(self.title_select_action)
        tool_group.addAction(self.title_select_action)

        toolbar.addSeparator()

        # --- NEW: Show Masks Action ---
        self.show_masks_action = QAction(style.standardIcon(style.SP_CustomBase), "Show Masks", self)
        self.show_masks_action.setCheckable(True)
        self.show_masks_action.setEnabled(False)
        self.show_masks_action.toggled.connect(self.toggle_masks_display)
        toolbar.addAction(self.show_masks_action)

        toolbar.addSeparator()

        # Undo/Redo Actions
        self.undo_action = QAction(style.standardIcon(style.SP_ArrowBack), "Undo", self)
        self.undo_action.triggered.connect(self.image_viewer.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction(style.standardIcon(style.SP_ArrowForward), "Redo", self)
        self.redo_action.triggered.connect(self.image_viewer.redo)
        toolbar.addAction(self.redo_action)

    def build_correction_tab(self):
        self.correction_tab = QWidget()
        self.tabs.addTab(self.correction_tab, "Corrections")
        correction_layout = QVBoxLayout(self.correction_tab)

        ocr_group = QGroupBox("OCR & Axis Corrections")
        ocr_layout = QFormLayout(ocr_group)
        self.plot_title_edit = QLineEdit()
        self.x_title_edit = QLineEdit()
        self.y_title_edit = QLineEdit()
        ocr_layout.addRow("Plot Title:", self.plot_title_edit)
        ocr_layout.addRow("X-Axis Title:", self.x_title_edit)
        ocr_layout.addRow("Y-Axis Title:", self.y_title_edit)

        self.ticks_table = QTableWidget()
        self.ticks_table.setColumnCount(3)
        self.ticks_table.setHorizontalHeaderLabels(["Tick Image", "Predicted Text", "Corrected Value"])
        self.ticks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.ticks_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.ticks_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        ocr_layout.addRow(self.ticks_table)

        series_group = QGroupBox("Data Series")
        series_layout = QVBoxLayout(series_group)
        self.series_scroll_area = QScrollArea()
        self.series_scroll_area.setWidgetResizable(True)
        self.series_widget_container = QWidget()
        self.series_container_layout = QVBoxLayout(self.series_widget_container)
        self.series_scroll_area.setWidget(self.series_widget_container)
        series_layout.addWidget(self.series_scroll_area)
 
        # --- NEW: Interpolation Controls ---
        interpolation_group = QGroupBox("Recalculation Options")
        interpolation_layout = QFormLayout(interpolation_group)
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Linear", "Cubic Spline", "None (Raw Points)"])
        interpolation_layout.addRow("Interpolation Method:", self.interpolation_combo)
 
        self.apply_corrections_btn = QPushButton(" Apply All Corrections & Update Plot")
        self.apply_corrections_btn.setIcon(self.style().standardIcon(self.style().SP_DialogApplyButton))
        self.apply_corrections_btn.setEnabled(False)

        correction_layout.addWidget(ocr_group)
        correction_layout.addWidget(series_group)
        correction_layout.addWidget(interpolation_group)
        correction_layout.addWidget(self.apply_corrections_btn)

    def build_results_tab(self):
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results")
        results_layout = QVBoxLayout(self.results_tab)

        self.plot_tab = QLabel("Recreated plot will appear here.")
        self.plot_tab.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.plot_tab)

    def init_models(self):
        self.statusBar().showMessage("Loading models... Please wait.")
        self.extractor = ChartExtractor()
        self.statusBar().showMessage("Models loaded successfully.", 5000)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path)
            self.image_viewer.set_image(pixmap)
            self.run_btn.setEnabled(True)
            self.clear_results(clear_image_viewer_state=False)

    def run_inference(self):
        if self.image_viewer.editable_pixmap is None: return
        self.statusBar().showMessage("Running inference... This may take a while.")
        self.run_btn.setEnabled(False)
        self.apply_corrections_btn.setEnabled(False)
        self.show_masks_action.setEnabled(False) # Disable while running
        edited_image_np = self.image_viewer.get_edited_image_as_numpy()

        if edited_image_np is None:
            QMessageBox.critical(self, "Image Error", "Could not process the image. Please try loading it again.")
            self.run_btn.setEnabled(True)
            return

        self.worker = Worker(self.extractor, edited_image_np)
        self.worker.finished.connect(self.on_inference_complete)
        self.worker.start()

    def on_inference_complete(self, result):
        self.statusBar().showMessage(result['message'], 5000)
        self.run_btn.setEnabled(True)
        if result['status'] == 'success':
            self.current_results = result
            self.mask_overlay_pixmap = None # Clear cached overlay
            self.populate_correction_fields(result)
            self.update_plot(result)
            self.apply_corrections_btn.setEnabled(True)
            self.export_csv_action.setEnabled(True)
            self.export_mat_action.setEnabled(SCIPY_AVAILABLE)
            self.show_masks_action.setEnabled(True) # Enable mask button
        else:
            QMessageBox.critical(self, "Inference Error", result['message'])

    def populate_correction_fields(self, result):
        self.clear_results(clear_image_viewer_state=False) # Keep image viewer state
        ocr_data = result.get('ocr_data', {})
        self.plot_title_edit.setText(ocr_data.get('plot_title', {}).get('text', ''))
        self.x_title_edit.setText(ocr_data.get('x_axis_title', {}).get('text', ''))
        self.y_title_edit.setText(ocr_data.get('y_axis_title', {}).get('text', ''))

        self.ticks_table.setRowCount(0)
        for tick in ocr_data.get('ticks', []):
            row = self.ticks_table.rowCount()
            self.ticks_table.insertRow(row)

            crop_image_np = tick.get('crop_image')
            if crop_image_np is not None:
                if crop_image_np.shape[2] == 3:
                    bgra_image = cv2.cvtColor(crop_image_np, cv2.COLOR_BGR2BGRA)
                else:
                    bgra_image = crop_image_np

                h, w, ch = bgra_image.shape
                q_img = QImage(bgra_image.data, w, h, ch * w, QImage.Format_ARGB32)
                pixmap = QPixmap.fromImage(q_img)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaledToHeight(25, Qt.SmoothTransformation))
                image_label.setAlignment(Qt.AlignCenter)
                self.ticks_table.setCellWidget(row, 0, image_label)

            predicted_item = QTableWidgetItem(tick['text'])
            predicted_item.setFlags(predicted_item.flags() & ~Qt.ItemIsEditable)
            self.ticks_table.setItem(row, 1, predicted_item)
            self.ticks_table.setItem(row, 2, QTableWidgetItem(str(tick['value'])))

        self.ticks_table.resizeRowsToContents()

        for i, df in enumerate(result.get('dataframes', [])):
            series_name = result['series_data'][i]['series_name']
            editor = SeriesEditorWidget(series_name, df)
            self.series_container_layout.addWidget(editor)
            self.series_widgets.append(editor)

    def apply_all_corrections(self):
        if not self.current_results: return

        corrected_ocr = self.current_results['ocr_data']
        corrected_ocr['plot_title']['text'] = self.plot_title_edit.text()
        corrected_ocr['x_axis_title']['text'] = self.x_title_edit.text()
        corrected_ocr['y_axis_title']['text'] = self.y_title_edit.text()
        for i in range(self.ticks_table.rowCount()):
            try:
                corrected_ocr['ticks'][i]['value'] = float(self.ticks_table.item(i, 2).text())
            except (ValueError, IndexError):
                pass

        # --- NEW: Get selected interpolation method ---
        inter_text = self.interpolation_combo.currentText()
        inter_map = {
            "Linear": "linear",
            "Cubic Spline": "cubic_spline",
            "None (Raw Points)": "none"
        }
        interpolation_method = inter_map.get(inter_text, 'linear')

        # --- MODIFIED: Use raw_keypoints from results ---
        filtered_lines, new_names = [], []
        for i, widget in enumerate(self.series_widgets):
            if not widget.is_deleted:
                # The key in current_results is now 'raw_keypoints'
                filtered_lines.append(self.current_results['raw_keypoints'][i])
                new_names.append(widget.get_series_name())

        # --- MODIFIED: Pass interpolation method to recalculate ---
        recalculated = self.extractor.recalculate_from_corrected(
            filtered_lines, corrected_ocr, interpolation_method=interpolation_method
        )

        if recalculated['status'] == 'success':
            for i, name in enumerate(new_names):
                if i < len(recalculated['series_data']):
                    recalculated['series_data'][i]['series_name'] = name
            
            # --- FIX: Merge new results into the existing state ---
            # The 'recalculated' dict from the backend only contains the new data points and plot titles.
            # We must merge this into `self.current_results` to preserve essential data
            # like 'raw_keypoints' and 'instance_masks' for subsequent corrections or for toggling the mask display.
            self.current_results.update(recalculated)
            # We also need to explicitly update 'ocr_data' in our state with the latest
            # corrections from the GUI, so the next "Apply" click uses the right values.
            self.current_results['ocr_data'] = corrected_ocr

            self.update_plot(self.current_results)
            self.statusBar().showMessage("Recalculated with all corrections.", 5000)
        else:
            QMessageBox.warning(self, "Recalculation Error", recalculated['message'])

    def update_plot(self, result):
        # --- FIX: Add the shared colors to the result dictionary before plotting ---
        plot_colors = [color.name(QColor.HexRgb) for color in self.SERIES_COLORS]
        result_with_colors = result.copy()
        result_with_colors['colors'] = plot_colors

        plot_image_np = self.extractor._recreate_plot_image(result_with_colors)
        if plot_image_np is not None:
            h, w, ch = plot_image_np.shape
            q_img = QImage(plot_image_np.data, w, h, ch * w, QImage.Format_ARGB32)
            self.plot_tab.setPixmap(QPixmap.fromImage(q_img))
            self.tabs.setCurrentWidget(self.results_tab)

    def handle_manual_ocr(self, bbox):
        """Slot to handle OCR on a manually selected region."""
        if self.extractor is None or self.image_viewer.editable_pixmap is None:
            return

        image_np = self.image_viewer.get_edited_image_as_numpy()
        ocr_text = self.extractor.ocr_on_region(image_np, bbox)

        if ocr_text:
            self.plot_title_edit.setText(ocr_text)
            self.statusBar().showMessage(f"OCR result for title: '{ocr_text}'", 5000)
        else:
            self.statusBar().showMessage("No text found in the selected region.", 3000)

        self.brush_action.setChecked(True)
        self.image_viewer.set_tool('brush')

    def toggle_masks_display(self, checked):
        """Creates and shows/hides the segmentation mask overlay."""
        if not checked:
            self.image_viewer.show_masks(False)
            return

        if not self.current_results or 'instance_masks' not in self.current_results:
            return

        # Use cached overlay if it exists
        if self.mask_overlay_pixmap:
            self.image_viewer.set_mask_overlay(self.mask_overlay_pixmap)
            self.image_viewer.show_masks(True)
            return

        masks = self.current_results['instance_masks']
        if not masks:
            return

        # Create the overlay
        h, w = masks[0].shape
        overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

        for i, mask in enumerate(masks):
            # --- FIX: Use the shared color palette ---
            q_color = self.SERIES_COLORS[i % len(self.SERIES_COLORS)]
            # Get RGBA components for the numpy array
            color_rgba = (q_color.red(), q_color.green(), q_color.blue(), q_color.alpha())

            boolean_mask = mask == 255
            # OpenCV uses BGRA, so we need to construct the color tuple accordingly
            overlay_np[boolean_mask] = (color_rgba[2], color_rgba[1], color_rgba[0], color_rgba[3])

        # Convert to QPixmap
        q_img = QImage(overlay_np.data, w, h, w * 4, QImage.Format_ARGB32)
        self.mask_overlay_pixmap = QPixmap.fromImage(q_img)

        # Set and show in the viewer
        self.image_viewer.set_mask_overlay(self.mask_overlay_pixmap)
        self.image_viewer.show_masks(True)

    def export_as_csv(self):
        if not self.current_results or 'dataframes' not in self.current_results:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save as CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                all_dfs = []
                for i, series in enumerate(self.current_results['series_data']):
                    df = pd.DataFrame(series['data_points'])
                    series_name = series.get('series_name', f'series_{i + 1}')
                    df.columns = [f'{series_name}_x', f'{series_name}_y']
                    all_dfs.append(df)

                combined_df = pd.concat(all_dfs, axis=1)
                combined_df.to_csv(path, index=False)
                self.statusBar().showMessage(f"Data exported to {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def export_as_mat(self):
        if not self.current_results or 'dataframes' not in self.current_results:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save as .mat", "", "MATLAB Files (*.mat)")
        if path:
            try:
                mat_dict = {
                    'plot_title': self.current_results.get('plot_title', ''),
                    'x_axis_title': self.current_results.get('x_axis_title', ''),
                    'y_axis_title': self.current_results.get('y_axis_title', '')
                }
                for series in self.current_results['series_data']:
                    series_name = series.get('series_name', 'unnamed_series').replace(' ', '_').replace('(', '').replace(
                        ')', '')
                    df = pd.DataFrame(series['data_points'])
                    mat_dict[series_name] = df.to_numpy()

                savemat(path, mat_dict)
                self.statusBar().showMessage(f"Data exported to {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export .mat file: {e}")

    def clear_results(self, clear_image_viewer_state=True):
        self.plot_title_edit.clear()
        self.x_title_edit.clear()
        self.y_title_edit.clear()
        self.ticks_table.setRowCount(0)
        for widget in self.series_widgets:
            widget.deleteLater()
        self.series_widgets = []
        self.plot_tab.setText("Recreated plot will appear here.")
        self.apply_corrections_btn.setEnabled(False)
        self.export_csv_action.setEnabled(False)
        self.export_mat_action.setEnabled(False)
        self.show_masks_action.setEnabled(False)
        self.show_masks_action.setChecked(False)
        self.mask_overlay_pixmap = None
        if clear_image_viewer_state:
            self.image_viewer.set_image(QPixmap()) # Clear image viewer

