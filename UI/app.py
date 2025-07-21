# main_gui.py
import pandas as pd
import copy
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QPushButton, QFileDialog, QLabel, QLineEdit, QFormLayout, QComboBox, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QMessageBox,
                             QGroupBox, QScrollArea, QAction, QToolBar, QActionGroup, QDoubleSpinBox, QSpinBox,
                             QToolButton, QMenu)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from model.chart_extractor import ChartExtractor
from model.pdf_chart_extractor import PdfChartExtractor
from model.post_processor import PostProcessor
from .interactive_image_viewer import InteractiveImageViewer
from .gui_widgets import (SeriesEditorWidget, OcrDebugWidget, InteractivePlotCanvas,
                          CorrectionTab, PostProcessingTab, SaveTab, ResultsTab)
from .chart_selection_dialog import ChartSelectionDialog

# Try to import scipy for .mat export, handle if not found
try:
    from scipy.io import savemat

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Worker(QThread):
    """Worker thread for running heavy model inference."""
    finished = pyqtSignal(dict)

    def __init__(self, extractor, image_data, ocr_params=None):
        super().__init__()
        self.extractor = extractor
        self.image_data = image_data
        self.ocr_params = ocr_params

    def run(self):
        # Pass cached ocr_params if they exist to skip auto-tuning
        result = self.extractor.process_image(self.image_data, ocr_params=self.ocr_params)
        self.finished.emit(result)


class PdfWorker(QThread):
    """Worker thread for extracting charts from a PDF."""
    finished = pyqtSignal(list)

    def __init__(self, pdf_extractor, pdf_path):
        super().__init__()
        self.pdf_extractor = pdf_extractor
        self.pdf_path = pdf_path

    def run(self):
        # This can be a long process
        extracted_charts = self.pdf_extractor.extract_charts_from_pdf(self.pdf_path)
        self.finished.emit(extracted_charts)


class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chart Data Extractor Ver 1.0")
        self.setGeometry(100, 100, 1600, 900)
        self.extractor = None
        self.current_results = None
        self.pdf_extractor = None  # New extractor instance
        self.series_widgets = []
        self.ocr_debug_widgets = []
        self.processed_results = None
        self.ocr_target_widget = None
        self.best_ocr_params = {}
        self.mask_overlay_pixmap = None  # Cache for the overlay
        self.chart_selection_dialog = None  # To hold reference to the dialog

        # --- NEW: Define a shared color palette for masks and plots ---
        self.SERIES_COLORS = [
            QColor(31, 119, 180, 120), QColor(255, 127, 14, 120), QColor(44, 160, 44, 120),
            QColor(214, 39, 40, 120), QColor(148, 103, 189, 120), QColor(140, 86, 75, 120)
        ]  # R, G, B, Alpha (for overlay)

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # --- NEW: Use QSplitter for resizable panels ---
        splitter = QSplitter(Qt.Horizontal)

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
        self.correction_tab = CorrectionTab()
        self.results_tab = ResultsTab()
        self.postprocessing_tab = PostProcessingTab()
        self.save_tab = SaveTab()
        self.ocr_debug_tab = QWidget()  # Keep OCR debug tab simple for now

        self.tabs.addTab(self.correction_tab, "Corrections")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.postprocessing_tab, "Post-processing")
        self.tabs.addTab(self.save_tab, "Save & Export")
        self.tabs.addTab(self.ocr_debug_tab, "OCR Debug")

        self.build_ocr_debug_tab()  # This method will now just populate the ocr_debug_tab

        right_panel_layout.addWidget(controls_group)
        right_panel_layout.addWidget(self.tabs)

        # --- NEW: Add panels to splitter and set main layout ---
        splitter.addWidget(left_panel_group)
        splitter.addWidget(right_panel_widget)
        splitter.setSizes([1000, 600])  # Set initial proportional sizes

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Use full space
        main_layout.addWidget(splitter)

        # --- Connections ---
        self.run_btn.clicked.connect(self.run_inference)
        self.correction_tab.apply_corrections_btn.clicked.connect(self.apply_all_corrections)
        self.postprocessing_tab.apply_postprocessing_btn.clicked.connect(self.handle_apply_postprocessing)
        self.postprocessing_tab.reset_series_btn.clicked.connect(self.handle_reset_series)
        self.postprocessing_tab.reset_all_btn.clicked.connect(self.handle_reset_all_series)
        self.postprocessing_tab.postproc_series_combo.currentIndexChanged.connect(self._display_current_postproc_state)
        self.image_viewer.regionSelected.connect(self.handle_region_ocr)
        self.postprocessing_tab.postproc_canvas.points_updated.connect(self.handle_interactive_plot_changed)

        # --- Connect post-processing controls to their handlers ---
        # Outlier controls
        self.postprocessing_tab.outlier_enabled_check.toggled.connect(self.toggle_outlier_controls)
        self.postprocessing_tab.outlier_method_combo.currentTextChanged.connect(self.handle_outlier_method_change)
        self.postprocessing_tab.outlier_window_spin.valueChanged.connect(self._run_postprocessing_preview)
        self.postprocessing_tab.outlier_threshold_spin.valueChanged.connect(self._run_postprocessing_preview)
        # Auto-resampling controls
        self.postprocessing_tab.auto_resampling_enabled_check.toggled.connect(self._update_postproc_controls_state)
        self.postprocessing_tab.auto_resample_points_spin.valueChanged.connect(self._run_postprocessing_preview)
        self.postprocessing_tab.auto_resample_method_combo.currentTextChanged.connect(self._run_postprocessing_preview)
        # Manual editing controls
        self.postprocessing_tab.manual_editing_enabled_check.toggled.connect(self._update_postproc_controls_state)
        self.postprocessing_tab.manual_init_reset_btn.clicked.connect(self.handle_init_reset_manual_points)
        self.postprocessing_tab.manual_final_points_spin.valueChanged.connect(self._run_postprocessing_preview)
        self.postprocessing_tab.manual_interp_method_combo.currentTextChanged.connect(self._run_postprocessing_preview)
        # View options
        self.postprocessing_tab.show_original_check.toggled.connect(self._display_current_postproc_state)

        self.correction_tab.plot_title_ocr_btn.clicked.connect(
            lambda: self.start_ocr_selection(self.correction_tab.plot_title_edit, "Plot Title", rotate=False))
        self.correction_tab.x_title_ocr_btn.clicked.connect(
            lambda: self.start_ocr_selection(self.correction_tab.x_title_edit, "X-Axis Title", rotate=False))
        self.correction_tab.y_title_ocr_btn.clicked.connect(
            lambda: self.start_ocr_selection(self.correction_tab.y_title_edit, "Y-Axis Title", rotate=True))

        self.save_tab.export_csv_btn.clicked.connect(self.export_as_csv)
        self.save_tab.export_mat_btn.clicked.connect(self.export_as_mat)

        # Set initial state of controls
        self._update_postproc_controls_state()

        self.init_models()

    def build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        style = self.style()

        # File Actions
        self.load_action = QAction(style.standardIcon(style.SP_DirOpenIcon), "Open Image", self)
        self.load_action.triggered.connect(self.load_image)
        toolbar.addAction(self.load_action)

        self.load_pdf_action = QAction(style.standardIcon(style.SP_FileIcon), "Open PDF", self)
        self.load_pdf_action.triggered.connect(self.load_pdf)
        toolbar.addAction(self.load_pdf_action)

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

        toolbar.addSeparator()

        # --- NEW: Show Masks Action ---
        self.show_masks_action = QAction(style.standardIcon(style.SP_CustomBase), "Show Masks", self)
        self.show_masks_action.setCheckable(True)
        self.show_masks_action.setEnabled(False)
        self.show_masks_action.toggled.connect(self.toggle_masks_display)
        toolbar.addAction(self.show_masks_action)

        # --- NEW: Correct Ticks Action ---
        self.correct_ticks_action = QAction(style.standardIcon(style.SP_CustomBase), "Correct Tick Positions", self)
        self.correct_ticks_action.setCheckable(True)
        self.correct_ticks_action.setEnabled(False)
        self.correct_ticks_action.toggled.connect(self.toggle_draggable_ticks_display)
        toolbar.addAction(self.correct_ticks_action)

        toolbar.addSeparator()

        # Undo/Redo Actions
        self.undo_action = QAction(style.standardIcon(style.SP_ArrowBack), "Undo", self)
        self.undo_action.triggered.connect(self.image_viewer.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction(style.standardIcon(style.SP_ArrowForward), "Redo", self)
        self.redo_action.triggered.connect(self.image_viewer.redo)
        toolbar.addAction(self.redo_action)

    def _update_all_views(self):
        """Central function to refresh all data-dependent UI components."""
        # Determine the final data to be displayed and saved
        final_data_source = self.processed_results if self.processed_results else self.current_results

        if not final_data_source or not [s for s in final_data_source.get('series_data', []) if
                                         not s.get('is_deleted', False)]:
            # Clear all views if there is no data
            self.results_tab.plot_label.setText("Recreated plot will appear here.")
            self.postprocessing_tab.postproc_canvas.update_plot([], self.SERIES_COLORS)
            self.save_tab.save_preview_table.setRowCount(0)
            self.save_tab.save_preview_table.setColumnCount(0)
            self.save_tab.export_csv_btn.setEnabled(False)
            self.save_tab.export_mat_btn.setEnabled(False)
            return

        # --- NEW: Filter out deleted series before displaying ---
        active_series_data = [s for s in final_data_source['series_data'] if not s.get('is_deleted', False)]
        final_data = final_data_source.copy()
        final_data['series_data'] = active_series_data

        # 1. Update the main recreated plot in the "Results" tab
        self.update_plot(final_data)

        # 2. Update the post-processing plot
        self._display_current_postproc_state()

        # 3. Update the save preview table
        self._update_save_preview_table(final_data)

        # 4. Enable export buttons
        self.save_tab.export_csv_btn.setEnabled(True)
        self.save_tab.export_mat_btn.setEnabled(SCIPY_AVAILABLE)

    def _update_save_preview_table(self, results_data):
        """Populates the table in the Save tab with the final data."""
        self.save_tab.save_preview_table.setRowCount(0)
        self.save_tab.save_preview_table.setColumnCount(0)

        if not results_data or not results_data.get('series_data'):
            return

        # Consolidate data into a single DataFrame for preview
        all_dfs = []
        for i, series in enumerate(results_data['series_data']):
            df = pd.DataFrame(series['data_points'])
            series_name = series.get('series_name', f'series_{i + 1}')
            # --- FIX: Check for empty dataframe before assigning columns ---
            if df.empty:
                # For empty series, create an empty DataFrame with the correct column names
                # to prevent errors and ensure proper alignment during concatenation.
                df = pd.DataFrame(columns=[f'{series_name}_x', f'{series_name}_y'])
            else:
                df.columns = [f'{series_name}_x', f'{series_name}_y']
            all_dfs.append(df)

        if not all_dfs:
            return

        combined_df = pd.concat(all_dfs, axis=1)

        # Populate the QTableWidget
        self.save_tab.save_preview_table.setColumnCount(len(combined_df.columns))
        self.save_tab.save_preview_table.setHorizontalHeaderLabels(combined_df.columns)
        self.save_tab.save_preview_table.setRowCount(len(combined_df))

        for i, row in combined_df.iterrows():
            for j, col in enumerate(combined_df.columns):
                item = QTableWidgetItem(str(row[col]))
                self.save_tab.save_preview_table.setItem(i, j, item)

        self.save_tab.save_preview_table.resizeColumnsToContents()

    def build_ocr_debug_tab(self):
        layout = QVBoxLayout(self.ocr_debug_tab)

        # --- NEW: Global OCR Controls ---
        global_controls_group = QGroupBox("Global OCR Parameters")
        global_controls_layout = QFormLayout(global_controls_group)

        self.ocr_scale_spinbox = QDoubleSpinBox()
        self.ocr_scale_spinbox.setRange(0.5, 5.0)
        self.ocr_scale_spinbox.setSingleStep(0.1)
        self.ocr_scale_spinbox.setValue(2.0)
        global_controls_layout.addRow("Scale Factor:", self.ocr_scale_spinbox)

        self.ocr_blur_spinbox = QSpinBox()
        self.ocr_blur_spinbox.setRange(1, 21)
        self.ocr_blur_spinbox.setSingleStep(2)
        self.ocr_blur_spinbox.setValue(5)
        global_controls_layout.addRow("Blur Kernel Size:", self.ocr_blur_spinbox)

        self.ocr_margin_spinbox = QSpinBox()
        self.ocr_margin_spinbox.setRange(0, 20)
        self.ocr_margin_spinbox.setValue(2)
        global_controls_layout.addRow("Margin (px):", self.ocr_margin_spinbox)

        self.rerun_all_ocr_btn = QPushButton("Re-run All OCR with these Parameters")
        self.rerun_all_ocr_btn.clicked.connect(self.handle_global_ocr_rerun)
        self.rerun_all_ocr_btn.setEnabled(False)

        layout.addWidget(global_controls_group)
        layout.addWidget(self.rerun_all_ocr_btn)
        # --- End of Global Controls ---

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.ocr_debug_container = QWidget()
        self.ocr_debug_layout = QVBoxLayout(self.ocr_debug_container)
        # Align widgets to the top, so they don't spread out vertically
        self.ocr_debug_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.ocr_debug_container)

        layout.addWidget(scroll_area)

    def init_models(self):
        self.statusBar().showMessage("Loading models... Please wait.")
        # --- NEW: Initialize both extractors ---
        self.extractor = ChartExtractor()
        self.pdf_extractor = PdfChartExtractor()
        self.statusBar().showMessage("Models loaded successfully.", 5000)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path)
            self.image_viewer.set_image(pixmap)
            self.run_btn.setEnabled(True)
            self.clear_results(clear_image_viewer_state=False)
            # Reset cached OCR params for a new image to force re-tuning.
            self.best_ocr_params = {}

    def load_pdf(self):
        """Opens a PDF and starts a worker to extract chart images."""
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if path:
            self.statusBar().showMessage("Processing PDF to find charts... This may take a while.", 0)
            # Disable buttons to prevent user interaction during processing
            self.load_action.setEnabled(False)
            self.load_pdf_action.setEnabled(False)
            self.run_btn.setEnabled(False)

            self.pdf_worker = PdfWorker(self.pdf_extractor, path)
            self.pdf_worker.finished.connect(self.on_pdf_processing_complete)
            self.pdf_worker.start()

    def on_pdf_processing_complete(self, chart_images):
        """
        Receives extracted charts from the PDF worker and opens the selection dialog.
        """
        # Re-enable buttons
        self.load_action.setEnabled(True)
        self.load_pdf_action.setEnabled(True)
        self.statusBar().clearMessage()

        self.chart_selection_dialog = ChartSelectionDialog(chart_images, self)
        self.chart_selection_dialog.chartSelected.connect(self.load_selected_chart_from_dialog)
        self.chart_selection_dialog.show()

    def load_selected_chart_from_dialog(self, pixmap):
        """This slot receives the selected chart from the dialog and loads it."""
        if pixmap and not pixmap.isNull():
            self.image_viewer.set_image(pixmap)
            self.clear_results(clear_image_viewer_state=False)
            self.run_btn.setEnabled(True)
            self.best_ocr_params = {}  # Reset for new image
            self.statusBar().showMessage("Selected chart loaded. You can now run inference.", 5000)

    def run_inference(self):
        if self.image_viewer.editable_pixmap is None: return
        self.statusBar().showMessage("Running inference... This may take a while.")
        self.run_btn.setEnabled(False)
        self.correction_tab.apply_corrections_btn.setEnabled(False)
        self.show_masks_action.setEnabled(False)  # Disable while running
        self.correct_ticks_action.setEnabled(False)  # Disable while running
        edited_image_np = self.image_viewer.get_edited_image_as_numpy()

        if edited_image_np is None:
            QMessageBox.critical(self, "Image Error", "Could not process the image. Please try loading it again.")
            self.run_btn.setEnabled(True)
            return

        # Pass cached OCR params to the worker to skip auto-tuning on re-runs
        self.worker = Worker(self.extractor, edited_image_np, ocr_params=self.best_ocr_params)
        self.worker.finished.connect(self.on_inference_complete)
        self.worker.start()

    def on_inference_complete(self, result):
        self.statusBar().showMessage(result.get('message', 'Processing complete.'), 5000)
        self.run_btn.setEnabled(True)
        self.current_results = result
        self.correct_ticks_action.setEnabled(False)  # Disable before checks

        # --- REFACTORED WORKFLOW ---
        # 1. Clear all results from the previous run first.
        self.clear_results(clear_image_viewer_state=False, clear_ocr_debug=True)

        # 2. Always populate the OCR debug tab if data is available, regardless of success.
        if result.get('ocr_data'):
            self.populate_ocr_debug_tab(result)
            self.rerun_all_ocr_btn.setEnabled(True)
        else:
            self.rerun_all_ocr_btn.setEnabled(False)

        # 3. Populate other tabs and enable features only on success.
        if result['status'] == 'success':
            self.mask_overlay_pixmap = None  # Clear cached overlay
            # --- NEW: Save best OCR params for later use ---
            self.best_ocr_params = result.get('ocr_data', {}).get('best_params_found', {})

            # --- NEW: Display the found best params in the OCR debug tab ---
            if self.best_ocr_params:
                self.ocr_scale_spinbox.setValue(self.best_ocr_params.get('scale_factor', 2.0))
                self.ocr_blur_spinbox.setValue(self.best_ocr_params.get('blur_ksize', 5))
                self.ocr_margin_spinbox.setValue(self.best_ocr_params.get('margin', 2))

            self.populate_correction_fields(result)
            self._update_postproc_combo_box()  # Centralized update
            self._update_all_views()  # Central update
            self.correction_tab.apply_corrections_btn.setEnabled(True)
            self.show_masks_action.setEnabled(True)
            self.correct_ticks_action.setEnabled(True)  # Enable on success
            self.postprocessing_tab.reset_all_btn.setEnabled(False)  # Disabled until first processing
            self.postprocessing_tab.apply_postprocessing_btn.setEnabled(True)
            # Clear the plot with default linear scale until user interacts.
            self.postprocessing_tab.postproc_canvas.update_plot([], [], x_scale='linear', y_scale='linear')
        else:
            # On failure, the OCR debug tab is still populated, so we just show the error.
            self.correct_ticks_action.setEnabled(False)
            QMessageBox.critical(self, "Inference Error", result['message'])
            self.postprocessing_tab.apply_postprocessing_btn.setEnabled(False)

    def populate_correction_fields(self, result):
        # --- FIX: This method should only populate, not clear. Clearing is now handled by the caller. ---
        ocr_data = result.get('ocr_data', {})

        # --- REFACTORED: Use helper to populate OCR fields ---
        self._update_correction_fields_from_data(ocr_data)

        for i, df in enumerate(result.get('dataframes', [])):
            series_info = result['series_data'][i]
            color = self.SERIES_COLORS[i % len(self.SERIES_COLORS)]
            editor = SeriesEditorWidget(series_info['series_name'], color, i)  # Pass original index
            self.correction_tab.series_container_layout.addWidget(editor)
            self.series_widgets.append(editor)
            # --- NEW: Connect to the centralized handler ---
            editor.nameChanged.connect(self.handle_series_metadata_changed)
            editor.deleteStateChanged.connect(self.handle_series_metadata_changed)
            editor.duplicateRequested.connect(lambda w=editor: self.handle_duplicate_series(w))
            editor.ocrRequested.connect(
                lambda w=editor: self.start_ocr_selection(w.name_edit, f"Series '{w.original_name}'", rotate=False))

    def _update_correction_fields_from_data(self, ocr_data):
        """
        Helper function to populate the OCR/Axis correction fields from an ocr_data dictionary.
        This is used for both initial population and after re-running global OCR.
        """
        self.correction_tab.plot_title_edit.setText(ocr_data.get('plot_title', {}).get('text', ''))
        self.correction_tab.x_title_edit.setText(ocr_data.get('x_axis_title', {}).get('text', ''))
        self.correction_tab.y_title_edit.setText(ocr_data.get('y_axis_title', {}).get('text', ''))

        self.correction_tab.x_ticks_table.setRowCount(0)
        self.correction_tab.y_ticks_table.setRowCount(0)

        for tick in ocr_data.get('ticks', []):
            # Decide which table to add the row to
            target_table = None
            if tick.get('axis') == 'x':
                target_table = self.correction_tab.x_ticks_table
            elif tick.get('axis') == 'y':
                target_table = self.correction_tab.y_ticks_table

            if target_table is None:
                continue

            row = target_table.rowCount()
            target_table.insertRow(row)

            crop_image_np = tick.get('crop_image')
            if crop_image_np is not None:
                if len(crop_image_np.shape) == 3 and crop_image_np.shape[2] == 3:
                    # Convert BGR to BGRA for QImage
                    bgra_image = cv2.cvtColor(crop_image_np, cv2.COLOR_BGR2BGRA)
                else:
                    # Assume it's already in a displayable format (e.g., BGRA or Grayscale)
                    bgra_image = crop_image_np

                h, w, ch = bgra_image.shape
                bytes_per_line = ch * w
                q_img = QImage(bgra_image.data, w, h, bytes_per_line, QImage.Format_ARGB32)
                pixmap = QPixmap.fromImage(q_img)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaledToHeight(25, Qt.SmoothTransformation))
                image_label.setAlignment(Qt.AlignCenter)
                target_table.setCellWidget(row, 0, image_label)

            predicted_item = QTableWidgetItem(tick['text'])
            predicted_item.setFlags(predicted_item.flags() & ~Qt.ItemIsEditable)
            target_table.setItem(row, 1, predicted_item)
            target_table.setItem(row, 2, QTableWidgetItem(str(tick['value'])))

        self.correction_tab.x_ticks_table.resizeRowsToContents()
        self.correction_tab.y_ticks_table.resizeRowsToContents()

    def populate_ocr_debug_tab(self, result):
        # Clear previous widgets
        for widget in self.ocr_debug_widgets:
            widget.deleteLater()
        self.ocr_debug_widgets = []

        ocr_debug_info = result.get('ocr_data', {}).get('ocr_debug_info', [])
        if not ocr_debug_info:
            label = QLabel("No OCR debug information available.")
            label.setAlignment(Qt.AlignCenter)
            self.ocr_debug_layout.addWidget(label)
            self.ocr_debug_widgets.append(label)
            return

        for info in ocr_debug_info:
            # Only add a widget if there was an actual image crop to process
            if info.get('original_crop') is not None:
                # --- MODIFIED: Use simplified widget, no signal connection ---
                widget = OcrDebugWidget(info)
                self.ocr_debug_layout.addWidget(widget)
                self.ocr_debug_widgets.append(widget)

    def apply_all_corrections(self):  # --- ENTIRELY REWRITTEN ---
        """
        Applies corrections from the UI. It intelligently determines if a full
        data recalculation is needed (if ticks changed) or if only a metadata
        update is required (if only titles changed).
        """
        if not self.current_results: return

        # --- 1. Create a working copy of OCR data and check for changes ---
        corrected_ocr = copy.deepcopy(self.current_results['ocr_data'])
        ticks_changed_by_position = False

        # --- Check for changes from draggable ticks ---
        if self.correct_ticks_action.isChecked():
            corrected_ticks_from_viewer = self.image_viewer.get_corrected_ticks()
            if len(corrected_ticks_from_viewer) == len(corrected_ocr['ticks']):
                for i, original_tick in enumerate(corrected_ocr['ticks']):
                    viewer_tick = corrected_ticks_from_viewer[i]
                    if original_tick['pixel_x'] != viewer_tick['pixel_x'] or \
                       original_tick['pixel_y'] != viewer_tick['pixel_y']:
                        ticks_changed_by_position = True
                    # Update the pixel coordinates in our working copy
                    original_tick['pixel_x'] = viewer_tick['pixel_x']
                    original_tick['pixel_y'] = viewer_tick['pixel_y']
            else:
                QMessageBox.critical(self, "Error", "Tick data mismatch. Cannot apply position corrections.")
                return

        def to_float_or_nan(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        # --- Check for changes from tick value tables ---
        new_x_tick_values = [to_float_or_nan(self.correction_tab.x_ticks_table.item(i, 2).text()) for i in
                             range(self.correction_tab.x_ticks_table.rowCount())]
        new_y_tick_values = [to_float_or_nan(self.correction_tab.y_ticks_table.item(i, 2).text()) for i in
                             range(self.correction_tab.y_ticks_table.rowCount())]

        old_x_tick_values = [t['value'] for t in self.current_results['ocr_data'].get('ticks', []) if t.get('axis') == 'x']
        old_y_tick_values = [t['value'] for t in self.current_results['ocr_data'].get('ticks', []) if t.get('axis') == 'y']

        # Compare tick values to see if a full recalculation is needed.
        ticks_changed_by_value = True
        if len(new_x_tick_values) == len(old_x_tick_values) and len(new_y_tick_values) == len(old_y_tick_values):
            if np.allclose(new_x_tick_values, old_x_tick_values, equal_nan=True) and \
                    np.allclose(new_y_tick_values, old_y_tick_values, equal_nan=True):
                ticks_changed_by_value = False

        corrected_plot_title = self.correction_tab.plot_title_edit.text()
        corrected_x_title = self.correction_tab.x_title_edit.text()
        corrected_y_title = self.correction_tab.y_title_edit.text()

        titles_changed = (corrected_plot_title != self.current_results['ocr_data'].get('plot_title', {}).get('text', '') or
                          corrected_x_title != self.current_results['ocr_data'].get('x_axis_title', {}).get('text', '') or
                          corrected_y_title != self.current_results['ocr_data'].get('y_axis_title', {}).get('text', ''))

        series_changed = (any(w.name_edit.text() != w.original_name or w.is_deleted for w in self.series_widgets))

        # --- 2. Decide on update path ---
        recalculation_needed = ticks_changed_by_position or ticks_changed_by_value
        if not recalculation_needed:
            # --- PATH A: Metadata-only update. No data recalculation needed. ---
            if not titles_changed:
                if not series_changed:
                    self.statusBar().showMessage("No changes to apply.", 3000)
                return

            print("--- Applying metadata-only corrections (titles) ---")
            # Update titles on both current and processed results
            for result_set in [self.current_results, self.processed_results]:
                if result_set:
                    result_set['plot_title'] = corrected_plot_title
                    result_set['x_axis_title'] = corrected_x_title
                    result_set['y_axis_title'] = corrected_y_title

            # Update the underlying OCR data as well for consistency
            self.current_results['ocr_data']['plot_title']['text'] = corrected_plot_title
            self.current_results['ocr_data']['x_axis_title']['text'] = corrected_x_title
            self.current_results['ocr_data']['y_axis_title']['text'] = corrected_y_title

            self._update_all_views()
            self.statusBar().showMessage("Chart titles updated successfully.", 5000)
            return

        # --- PATH B: Ticks changed. Full data recalculation is required. ---
        print("--- Tick positions or values changed, performing full recalculation ---")
        if self.processed_results:
            reply = QMessageBox.question(self, "Confirm Recalculation",
                                         "Changing axis tick values requires re-calculating all data points from scratch. "
                                         "This will discard any changes made in the 'Post-processing' tab.\n\n"
                                         "Do you want to continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Proceed with full recalculation
        corrected_ocr['plot_title']['text'] = corrected_plot_title
        corrected_ocr['x_axis_title']['text'] = corrected_x_title
        corrected_ocr['y_axis_title']['text'] = corrected_y_title

        # Update the tick values in the state with the new ones from the UI
        x_ticks_in_state = [t for t in corrected_ocr['ticks'] if t.get('axis') == 'x']
        for i, tick_widget_value in enumerate(new_x_tick_values):
            if i < len(x_ticks_in_state) and not np.isnan(tick_widget_value):
                x_ticks_in_state[i]['value'] = tick_widget_value

        y_ticks_in_state = [t for t in corrected_ocr['ticks'] if t.get('axis') == 'y']
        for i, tick_widget_value in enumerate(new_y_tick_values):
            if i < len(y_ticks_in_state) and not np.isnan(tick_widget_value):
                y_ticks_in_state[i]['value'] = tick_widget_value

        inter_text = self.correction_tab.interpolation_combo.currentText()
        inter_map = {"Linear": "linear", "Cubic Spline": "cubic_spline", "None (Raw Points)": "none"}
        interpolation_method = inter_map.get(inter_text, 'linear')

        lines_to_process = [(w.original_index, self.current_results['raw_keypoints'][w.original_index])
                            for w in self.series_widgets if not w.is_deleted]

        recalculated = self.extractor.recalculate_from_corrected(
            lines_to_process, corrected_ocr, interpolation_method=interpolation_method
        )

        if recalculated['status'] == 'success':
            was_processed = self.processed_results is not None
            if was_processed:
                self.processed_results = None

            for series in recalculated['series_data']:
                widget = next((w for w in self.series_widgets if w.original_index == series['original_index']), None)
                if widget:
                    series['series_name'] = widget.get_series_name()

            recalculated_map = {s['original_index']: s for s in recalculated['series_data']}
            for s in self.current_results['series_data']:
                if s['original_index'] in recalculated_map:
                    s['data_points'] = recalculated_map[s['original_index']]['data_points']
                    s['series_name'] = recalculated_map[s['original_index']]['series_name']

            self.current_results['dataframes'] = [pd.DataFrame(s['data_points']) for s in
                                                  self.current_results['series_data']]
            self.current_results['plot_title'] = recalculated.get('plot_title')
            self.current_results['x_axis_title'] = recalculated.get('x_axis_title')
            self.current_results['y_axis_title'] = recalculated.get('y_axis_title')
            self.current_results['x_scale_type'] = recalculated.get('x_scale_type')
            self.current_results['y_scale_type'] = recalculated.get('y_scale_type')
            self.current_results['ocr_data'] = corrected_ocr

            self._update_postproc_combo_box()
            self._update_all_views()
            if was_processed:
                self.statusBar().showMessage("Recalculated with corrections. Post-processing has been reset.", 6000)
            else:
                self.statusBar().showMessage("Recalculated with all corrections.", 5000)
        else:
            QMessageBox.warning(self, "Recalculation Error", recalculated['message'])

    def update_plot(self, result, target_label=None):
        # --- FIX: Add the shared colors to the result dictionary before plotting ---
        plot_colors = [color.name(QColor.HexRgb) for color in self.SERIES_COLORS]
        result_with_colors = result.copy()
        result_with_colors['colors'] = plot_colors

        plot_image_np = self.extractor._recreate_plot_image(result_with_colors)
        if plot_image_np is not None:
            h, w, ch = plot_image_np.shape
            q_img = QImage(plot_image_np.data, w, h, ch * w, QImage.Format_ARGB32)
            self.results_tab.plot_label.setPixmap(QPixmap.fromImage(q_img))

    def toggle_draggable_ticks_display(self, checked):
        """Shows or hides the draggable tick lines."""
        if not self.current_results or not self.current_results.get('ocr_data'):
            return

        if checked:
            ticks = self.current_results['ocr_data'].get('ticks', [])
            if not ticks:
                self.statusBar().showMessage("No ticks detected to correct.", 3000)
                self.correct_ticks_action.setChecked(False)
                return
            self.image_viewer.show_draggable_ticks(True, ticks_data=ticks)
            self.statusBar().showMessage("Tick correction mode enabled. Drag lines to adjust tick positions.", 5000)
        else:
            self.image_viewer.show_draggable_ticks(False)
            self.statusBar().clearMessage()

    def handle_apply_postprocessing(self):
        if not self.current_results or 'series_data' not in self.current_results:
            QMessageBox.warning(self, "No Data", "No data available to process.")
            return

        # Lazily initialize processed_results with a deep copy to avoid modifying original data
        if self.processed_results is None:
            self.processed_results = copy.deepcopy(self.current_results)

        self.statusBar().showMessage("Applying post-processing...", 2000)

        # 1. Gather parameters from the UI
        params = self._get_postprocessing_params()

        # 2. Get the data to process from the LATEST data (processed or original)
        # --- MODIFIED: Use the latest processed data as the source for applying changes. ---
        # This makes the "Apply" operation cumulative, matching the preview's behavior.
        source_data_for_apply = self.processed_results if self.processed_results else self.current_results
        selected_series_name = self.postprocessing_tab.postproc_series_combo.currentText()

        active_series = [s for s in source_data_for_apply['series_data'] if not s.get('is_deleted', False)]

        if selected_series_name == "All Series":
            series_to_process = active_series
        else:
            series_to_process = [s for s in active_series if s['series_name'] == selected_series_name]

        if not series_to_process:
            QMessageBox.warning(self, "No Series", "The selected series could not be found or is deleted.")
            return

        # 3. Call the backend post-processor
        override_points = None
        if selected_series_name != "All Series" and params.get('manual_editing_enabled', False):
            override_points = self.postprocessing_tab.postproc_canvas.interactive_points

        # --- NEW: Get scale types from the source data ---
        x_scale = source_data_for_apply.get('x_scale_type', 'linear')
        y_scale = source_data_for_apply.get('y_scale_type', 'linear')
        outlier_method = self._get_postprocessing_params().get('outlier_method')

        processed_chunk = PostProcessor.process(copy.deepcopy(series_to_process), params,
                                                override_points=override_points,
                                                x_scale=x_scale, y_scale=y_scale, outlier_method=outlier_method)

        # --- NEW: If manual editing was used, store the reference points for state persistence ---
        if override_points is not None:
            for series in processed_chunk:
                # Store a copy of the points that generated this version of the data
                series['reference_points'] = copy.deepcopy(override_points)

        # 4. Update ("commit") the processed_results state in place
        if selected_series_name == "All Series":
            self.processed_results['series_data'] = processed_chunk
        else:
            for i, s in enumerate(self.processed_results['series_data']):
                if s['series_name'] == selected_series_name:
                    self.processed_results['series_data'][i] = processed_chunk[0]
                    break

        self.processed_results['dataframes'] = [pd.DataFrame(s['data_points']) for s in
                                                self.processed_results['series_data']]

        # 5. Refresh all views to show the newly committed state
        self._update_all_views()

        self.tabs.setCurrentWidget(self.postprocessing_tab)
        self.postprocessing_tab.reset_all_btn.setEnabled(True)
        self.statusBar().showMessage(f"Changes for '{selected_series_name}' have been applied.", 5000)

    def _get_postprocessing_params(self):
        """Helper function to gather all post-processing parameters from the UI."""
        return {
            'outlier_removal_enabled': self.postprocessing_tab.outlier_enabled_check.isChecked(),
            'outlier_method': self.postprocessing_tab.outlier_method_combo.currentText().lower().replace(' ', '_'),
            'outlier_window_fraction': self.postprocessing_tab.outlier_window_spin.value(),
            'outlier_threshold': self.postprocessing_tab.outlier_threshold_spin.value(),

            'auto_resampling_enabled': self.postprocessing_tab.auto_resampling_enabled_check.isChecked(),
            'auto_resample_points': self.postprocessing_tab.auto_resample_points_spin.value(),
            'auto_resample_method': self.postprocessing_tab.auto_resample_method_combo.currentText().lower().replace(
                ' ', '_'),

            'manual_editing_enabled': self.postprocessing_tab.manual_editing_enabled_check.isChecked(),
            'manual_final_points': self.postprocessing_tab.manual_final_points_spin.value(),
            'manual_interp_method': self.postprocessing_tab.manual_interp_method_combo.currentText().lower().replace(
                ' ', '_'),
        }

    def _run_postprocessing_preview(self):
        """Shows a live preview of post-processing without committing the changes."""
        # Use the latest data available as the source for scales and series
        source_data = self.processed_results if self.processed_results else self.current_results

        if not source_data or not source_data.get('series_data'):
            self.postprocessing_tab.postproc_canvas.update_plot([], self.SERIES_COLORS, x_scale='linear',
                                                                y_scale='linear')
            return

        # Get latest parameters and scale types
        selected_series_name = self.postprocessing_tab.postproc_series_combo.currentText()
        params = self._get_postprocessing_params()
        x_scale = source_data.get('x_scale_type', 'linear')
        y_scale = source_data.get('y_scale_type', 'linear')
        outlier_method = params.get('outlier_method')

        # --- FIX: If all post-processing is disabled, show the last committed state ---
        # The user expects the plot to retain the processed data unless they explicitly
        # reset it. Re-running a "preview" with no filters active would misleadingly
        # show the original data. Instead, we call _display_current_postproc_state,
        # which correctly shows self.processed_results if it exists.
        is_any_processing_enabled = (
                params.get('outlier_removal_enabled') or
                params.get('auto_resampling_enabled') or
                params.get('manual_editing_enabled')
        )

        if not is_any_processing_enabled:
            self._display_current_postproc_state()
            return

        # --- MODIFIED: Use the latest processed data as the source for previews ---
        # This allows for chaining post-processing effects. If no post-processing has been
        # applied yet, it falls back to the original corrected data.

        # Determine the source series from the chosen data source
        if selected_series_name == "All Series":
            source_series_list = source_data.get('series_data', [])
        else:
            source_series = next(
                (s for s in source_data.get('series_data', []) if s['series_name'] == selected_series_name), None)
            source_series_list = [source_series] if source_series else []

        active_series = [s for s in source_series_list if not s.get('is_deleted', False)] if source_series_list else []

        if not active_series:
            self.postprocessing_tab.postproc_canvas.update_plot([], self.SERIES_COLORS, x_scale=x_scale,
                                                                y_scale=y_scale)
            return
        interactive_mode = (selected_series_name != "All Series" and params.get('manual_editing_enabled', False))

        # If in interactive mode, get the current points from the canvas to use for processing
        override_points = None
        if interactive_mode:
            override_points = self.postprocessing_tab.postproc_canvas.interactive_points
            # If we are in interactive mode but have no points yet (e.g. before initialization),
            # we can't process, so we just show an empty plot with interactive mode enabled.
            if not override_points:
                self.postprocessing_tab.postproc_canvas.update_plot([], self.SERIES_COLORS, interactive_mode=True,
                                                                    x_scale=x_scale, y_scale=y_scale)
                return

        # Run processing on a deep copy to not affect the original
        processed_preview = PostProcessor.process(copy.deepcopy(active_series), params,
                                                  override_points=override_points,
                                                  x_scale=x_scale, y_scale=y_scale, outlier_method=outlier_method)

        # --- NEW: Handle showing original data overlay ---
        original_data_to_show = None
        if self.postprocessing_tab.show_original_check.isChecked():
            original_data_to_show = active_series

        # Update the plot with the temporary result and optional original data
        self.postprocessing_tab.postproc_canvas.update_plot(
            processed_preview, self.SERIES_COLORS,
            original_series=original_data_to_show,
            interactive_mode=interactive_mode,
            x_scale=x_scale, y_scale=y_scale
        )

    def _display_current_postproc_state(self):
        """Displays the currently committed post-processing state in the plot."""
        source_data = self.processed_results if self.processed_results else self.current_results
        if not source_data or not source_data.get('series_data'):
            self.postprocessing_tab.postproc_canvas.update_plot([], self.SERIES_COLORS, x_scale='linear',
                                                                y_scale='linear')
            return

        all_active_series = [s for s in source_data['series_data'] if not s.get('is_deleted', False)]

        x_scale = source_data.get('x_scale_type', 'linear')
        y_scale = source_data.get('y_scale_type', 'linear')
        selected_series_name = self.postprocessing_tab.postproc_series_combo.currentText()

        series_to_plot = []
        original_series_to_show = None
        interactive_mode = False
        self.postprocessing_tab.postproc_canvas.interactive_points = []  # Clear old points before redrawing

        if selected_series_name == "All Series":
            series_to_plot = all_active_series
        else:
            series_to_plot = [s for s in all_active_series if s['series_name'] == selected_series_name]
            # If we are showing a single series, check if we should be in interactive mode
            if series_to_plot:
                interactive_mode = self.postprocessing_tab.manual_editing_enabled_check.isChecked()
                if interactive_mode:
                    # Restore the reference points from the processed data if they exist
                    ref_points = series_to_plot[0].get('reference_points')
                    if ref_points:
                        self.postprocessing_tab.postproc_canvas.interactive_points = ref_points

        # If the overlay is checked, we need to find the corresponding original data
        if self.postprocessing_tab.show_original_check.isChecked() and self.current_results:
            original_series_all = [s for s in self.current_results['series_data'] if not s.get('is_deleted', False)]
            if selected_series_name == "All Series":
                original_series_to_show = original_series_all
            else:
                original_series_to_show = [s for s in original_series_all if s['series_name'] == selected_series_name]

        self.postprocessing_tab.postproc_canvas.update_plot(
            series_to_plot, self.SERIES_COLORS,
            original_series=original_series_to_show,
            interactive_mode=interactive_mode,
            x_scale=x_scale, y_scale=y_scale
        )

    def _update_postproc_combo_box(self):
        """Keeps the post-processing series dropdown in sync with the series editor widgets."""
        current_selection = self.postprocessing_tab.postproc_series_combo.currentText()
        self.postprocessing_tab.postproc_series_combo.blockSignals(True)
        self.postprocessing_tab.postproc_series_combo.clear()
        self.postprocessing_tab.postproc_series_combo.addItem("All Series")
        new_items = [w.get_series_name() for w in self.series_widgets if not w.is_deleted]
        self.postprocessing_tab.postproc_series_combo.addItems(new_items)
        index = self.postprocessing_tab.postproc_series_combo.findText(current_selection)
        self.postprocessing_tab.postproc_series_combo.setCurrentIndex(index if index != -1 else 0)
        self.postprocessing_tab.postproc_series_combo.blockSignals(False)

    def handle_series_metadata_changed(self):
        """Handles changes to a series' name or deleted status."""
        for widget in self.series_widgets:
            series_name = widget.get_series_name()
            is_deleted = widget.is_deleted
            original_index = widget.original_index

            # Find the corresponding series in both result sets and update them
            if self.current_results:
                for series in self.current_results.get('series_data', []):
                    if series.get('original_index') == original_index:
                        series['series_name'] = series_name
                        series['is_deleted'] = is_deleted
                        break

            if self.processed_results:
                for series in self.processed_results.get('series_data', []):
                    if series.get('original_index') == original_index:
                        series['series_name'] = series_name
                        series['is_deleted'] = is_deleted
                        break

        self._update_postproc_combo_box()
        self._update_all_views()

    def handle_duplicate_series(self, source_widget):
        """Creates a new series by duplicating an existing one."""
        if not self.current_results or not self.current_results.get('series_data'):
            return

        source_index = source_widget.original_index

        # Find the source series to duplicate in the master list (current_results)
        source_series_current = next(
            (s for s in self.current_results['series_data'] if s.get('original_index') == source_index), None)
        if not source_series_current:
            QMessageBox.warning(self, "Error", "Could not find the source series to duplicate.")
            return

        # --- 1. Create a new unique index ---
        # Find the max existing index to ensure the new one is unique.
        max_index = max(s.get('original_index', -1) for s in self.current_results['series_data'])
        new_index = max_index + 1

        # --- 2. Duplicate the data in `current_results` ---
        new_current_series = copy.deepcopy(source_series_current)
        new_current_series['series_name'] = f"{source_series_current.get('series_name', 'series')} (copy)"
        new_current_series['original_index'] = new_index
        new_current_series.pop('reference_points', None)  # Cloned series shouldn't inherit reference points
        self.current_results['series_data'].append(new_current_series)
        self.current_results['dataframes'].append(pd.DataFrame(new_current_series['data_points']))

        # Also duplicate the underlying raw data. This is important for recalculation.
        if source_index < len(self.current_results.get('raw_keypoints', [])):
            self.current_results['raw_keypoints'].append(
                copy.deepcopy(self.current_results['raw_keypoints'][source_index]))
        if source_index < len(self.current_results.get('instance_masks', [])):
            self.current_results['instance_masks'].append(
                self.current_results['instance_masks'][source_index].copy())

        # --- 3. Duplicate the data in `processed_results` if it exists ---
        if self.processed_results:
            source_series_processed = next(
                (s for s in self.processed_results['series_data'] if s.get('original_index') == source_index), None)
            if source_series_processed:
                new_processed_series = copy.deepcopy(source_series_processed)
                new_processed_series['series_name'] = new_current_series['series_name']  # Keep name consistent
                new_processed_series['original_index'] = new_index
                new_processed_series.pop('reference_points', None)
                self.processed_results['series_data'].append(new_processed_series)
                self.processed_results['dataframes'].append(pd.DataFrame(new_processed_series['data_points']))

        # --- 4. Create and add the new UI widget ---
        color = self.SERIES_COLORS[new_index % len(self.SERIES_COLORS)]
        editor = SeriesEditorWidget(new_current_series['series_name'], color, new_index)
        self.correction_tab.series_container_layout.addWidget(editor)
        self.series_widgets.append(editor)

        # Connect its signals, including its own duplicate button
        editor.duplicateRequested.connect(lambda w=editor: self.handle_duplicate_series(w))
        editor.nameChanged.connect(self.handle_series_metadata_changed)
        editor.deleteStateChanged.connect(self.handle_series_metadata_changed)
        editor.ocrRequested.connect(
            lambda w=editor: self.start_ocr_selection(w.name_edit, f"Series '{w.original_name}'", rotate=False))

        # --- 5. Refresh UI ---
        self._update_postproc_combo_box()
        self._update_all_views()
        self.statusBar().showMessage(f"Series '{source_widget.get_series_name()}' duplicated.", 4000)

    def _update_postproc_controls_state(self, checked=None):
        """Manages the state of post-processing controls, ensuring mutual exclusivity."""
        sender = self.sender()

        # --- Mutual Exclusivity for Resampling Modes ---
        if sender == self.postprocessing_tab.auto_resampling_enabled_check and self.postprocessing_tab.auto_resampling_enabled_check.isChecked():
            self.postprocessing_tab.manual_editing_enabled_check.setChecked(False)
        elif sender == self.postprocessing_tab.manual_editing_enabled_check and self.postprocessing_tab.manual_editing_enabled_check.isChecked():
            self.postprocessing_tab.auto_resampling_enabled_check.setChecked(False)

        is_auto = self.postprocessing_tab.auto_resampling_enabled_check.isChecked()
        is_manual = self.postprocessing_tab.manual_editing_enabled_check.isChecked()

        # --- Enable/Disable Controls based on Checkboxes ---
        self.postprocessing_tab.auto_resample_points_spin.setEnabled(is_auto)
        self.postprocessing_tab.auto_resample_method_combo.setEnabled(is_auto)

        self.postprocessing_tab.manual_ref_points_spin.setEnabled(is_manual)
        self.postprocessing_tab.manual_init_reset_btn.setEnabled(is_manual)
        self.postprocessing_tab.manual_final_points_spin.setEnabled(is_manual)
        self.postprocessing_tab.manual_interp_method_combo.setEnabled(is_manual)

        # Outlier group is disabled entirely during manual editing
        self.postprocessing_tab.outlier_group.setEnabled(not is_manual)

        # Trigger a preview to reflect the new state
        self._run_postprocessing_preview()

    def toggle_outlier_controls(self, checked):
        """Enables/disables the child controls of the outlier group."""
        self.postprocessing_tab.outlier_method_combo.setEnabled(checked)
        self.postprocessing_tab.outlier_threshold_spin.setEnabled(checked)
        # Also handle the method-specific controls
        self.handle_outlier_method_change(self.postprocessing_tab.outlier_method_combo.currentText())
        self._run_postprocessing_preview()

    def handle_outlier_method_change(self, method_text):
        """Shows/hides controls specific to the selected outlier method."""
        is_local_regression = (method_text == "Local Regression")
        is_enabled = self.postprocessing_tab.outlier_enabled_check.isChecked()
        self.postprocessing_tab.outlier_window_spin.setVisible(is_local_regression)
        self.postprocessing_tab.outlier_window_label.setVisible(is_local_regression)
        self.postprocessing_tab.outlier_window_spin.setEnabled(is_local_regression and is_enabled)

    def handle_interactive_plot_changed(self, points):
        """Handles updates from the interactive plot when points are moved."""
        # --- MODIFIED: Since we only drag points, the number of points doesn't change.
        # We no longer need to update the spinbox. We just need to trigger the preview
        # to re-interpolate the line based on the new point positions.
        self._run_postprocessing_preview()

    def handle_init_reset_manual_points(self):
        """Initializes or resets the interactive points for manual editing."""
        if not self.current_results or not self.current_results.get('series_data'): return
        selected_series_name = self.postprocessing_tab.postproc_series_combo.currentText()
        if selected_series_name == "All Series":
            QMessageBox.information(self, "Select Series", "Please select a single series to edit manually.")
            return
        source_series = next(
            (s for s in self.current_results['series_data'] if s['series_name'] == selected_series_name), None)
        if not source_series: return
        num_ref_points = self.postprocessing_tab.manual_ref_points_spin.value()
        initial_ref_points = PostProcessor.resample_series(source_series['data_points'], num_ref_points, 'linear')
        self.postprocessing_tab.postproc_canvas.interactive_points = initial_ref_points
        self._run_postprocessing_preview()

    def handle_reset_series(self):
        """Resets the selected series in the post-processing tab to its original state."""
        selected_series_name = self.postprocessing_tab.postproc_series_combo.currentText()
        if selected_series_name == "All Series" or not self.current_results or not self.processed_results:
            return

        series_to_reset = next(
            (s for s in self.processed_results['series_data'] if s['series_name'] == selected_series_name), None)
        if not series_to_reset:
            return

        original_index = series_to_reset.get('original_index')
        original_series = next(
            (s for s in self.current_results['series_data'] if s.get('original_index') == original_index), None)

        if not original_series:
            return

        # Find the index in the processed_results list to replace
        processed_index = -1
        for i, s in enumerate(self.processed_results['series_data']):
            if s.get('original_index') == original_index:
                processed_index = i
                break

        if processed_index != -1:
            self.processed_results['series_data'][processed_index] = copy.deepcopy(original_series)

        # Also update the corresponding dataframe to prevent inconsistencies
        if processed_index < len(self.processed_results.get('dataframes', [])):
            self.processed_results['dataframes'][processed_index] = pd.DataFrame(original_series['data_points'])

        self.statusBar().showMessage(f"Series '{selected_series_name}' has been reset.", 3000)
        self._update_all_views()

    def handle_reset_all_series(self):
        """Resets all post-processing changes for all series."""
        if self.processed_results is None:
            return

        reply = QMessageBox.question(self, 'Reset All Post-processing',
                                     "Are you sure you want to discard all post-processing changes for all series?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.processed_results = None
            self.statusBar().showMessage("All post-processing changes have been reset.", 3000)
            self._update_all_views()

    def handle_global_ocr_rerun(self):
        """Slot to re-run OCR on all items using the global parameters."""
        # --- FIX: Check for numpy array correctly to avoid ValueError ---
        # The expression `not numpy_array` is ambiguous. Check for `is None` instead.
        image_np = self.image_viewer.get_edited_image_as_numpy()
        if not self.current_results or image_np is None:
            self.statusBar().showMessage("No image or initial results to work with.", 3000)
            return

        self.statusBar().showMessage("Re-running all OCR with new parameters...")

        yolo_results = self.current_results.get('yolo_results')

        if image_np is None or yolo_results is None:
            self.statusBar().showMessage("Error: Missing image or YOLO results for OCR.", 3000)
            return

        params = {
            'scale_factor': self.ocr_scale_spinbox.value(),
            'blur_ksize': self.ocr_blur_spinbox.value(),
            'margin': self.ocr_margin_spinbox.value()
        }

        # Call the new backend method
        new_ocr_data = self.extractor.rerun_all_ocr(image_np, yolo_results, ocr_params=params)

        # Update the application state
        self.current_results['ocr_data'] = new_ocr_data

        # Update the UI widgets
        new_debug_info_list = new_ocr_data.get('ocr_debug_info', [])
        for i, widget in enumerate(self.ocr_debug_widgets):
            if i < len(new_debug_info_list):
                info = new_debug_info_list[i]
                widget.update_contents(info['text'], info['processed_img'])

        # --- NEW: Update the main correction fields with the new OCR data ---
        self._update_correction_fields_from_data(new_ocr_data)

        self.statusBar().showMessage("All OCR regions have been updated.", 5000)
        QMessageBox.information(self, "OCR Updated",
                                "OCR results have been updated and reflected in the 'Corrections' tab. "
                                "Please review and click 'Apply All Corrections' to see the effect on the final data.")

    def start_ocr_selection(self, target_widget, target_name, rotate=False):
        """Activates selection mode to perform OCR for a specific widget."""
        self.ocr_target_widget = target_widget
        self.ocr_target_rotate = rotate
        self.image_viewer.set_tool('select_title')  # Re-using the selection tool logic
        self.statusBar().showMessage(f"Select region for '{target_name}' by dragging the mouse. Right-click to cancel.",
                                     5000)

    def handle_region_ocr(self, bbox):
        """Generic slot to handle OCR on any manually selected region."""
        if self.ocr_target_widget is None:
            return

        image_np = self.image_viewer.get_edited_image_as_numpy()
        if image_np is None: return

        # Use the best parameters found during inference, or empty dict if none.
        ocr_text, _ = self.extractor.ocr_on_region(
            image_np, bbox, ocr_params=self.best_ocr_params, rotate=self.ocr_target_rotate
        )

        if ocr_text:
            self.ocr_target_widget.setText(ocr_text)
            self.statusBar().showMessage(f"OCR result: '{ocr_text}'", 5000)
        else:
            self.statusBar().showMessage("No text found in the selected region.", 3000)

        # Reset state
        self.ocr_target_widget = None
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
        results_to_export = self.processed_results if self.processed_results else self.current_results
        if not results_to_export or not results_to_export.get('series_data'):
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Data as CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                all_dfs = []
                for i, series in enumerate(results_to_export['series_data']):
                    df = pd.DataFrame(series['data_points'])
                    series_name = series.get('series_name', f'series_{i + 1}')
                    # --- FIX: Check for empty dataframe before assigning columns ---
                    if df.empty:
                        df = pd.DataFrame(columns=[f'{series_name}_x', f'{series_name}_y'])
                    else:
                        df.columns = [f'{series_name}_x', f'{series_name}_y']
                    all_dfs.append(df)

                combined_df = pd.concat(all_dfs, axis=1)
                combined_df.to_csv(path, index=False)
                self.statusBar().showMessage(f"Data exported to {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def export_as_mat(self):
        """Exports the final data (raw or processed) to a .mat file."""
        results_to_export = self.processed_results if self.processed_results else self.current_results

        if not results_to_export or not results_to_export.get('series_data'):
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save as .mat", "", "MATLAB Files (*.mat)")
        if not path:
            return

        try:
            # The main dictionary that will be saved to the .mat file.
            # It contains one key, 'chart_data', which holds the struct.
            mat_struct = {
                'plot_title': results_to_export.get('plot_title', ''),
                'x_axis_title': results_to_export.get('x_axis_title', ''),
                'y_axis_title': results_to_export.get('y_axis_title', ''),
                'series': {}  # This will hold the data for each series
            }

            # Filter for active series and sanitize names for MATLAB
            active_series = [s for s in results_to_export['series_data'] if not s.get('is_deleted', False)]

            for s in active_series:
                series_name = s.get('series_name', 'unnamed_series')
                # Sanitize the name to be a valid MATLAB struct field name
                sanitized_name = ''.join(c if c.isalnum() else '_' for c in series_name)
                if not sanitized_name or not sanitized_name[0].isalpha():
                    sanitized_name = 'series_' + sanitized_name

                df = pd.DataFrame(s['data_points'])
                if not df.empty:
                    # Create a dictionary for the series with 'x' and 'y' fields
                    series_struct = {
                        'x': df.iloc[:, 0].to_numpy(),
                        'y': df.iloc[:, 1].to_numpy()
                    }
                    mat_struct['series'][sanitized_name] = series_struct
                else:
                    mat_struct['series'][sanitized_name] = {'x': np.array([]), 'y': np.array([])}

            # The final dictionary to save. The variable name in MATLAB will be 'chart_data'.
            savemat(path, {'chart_data': mat_struct})
            self.statusBar().showMessage(f"Data exported to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export .mat file: {e}")

    def clear_results(self, clear_image_viewer_state=True, clear_ocr_debug=True):
        self.correction_tab.plot_title_edit.clear()
        self.correction_tab.x_title_edit.clear()
        self.correction_tab.y_title_edit.clear()
        self.correction_tab.x_ticks_table.setRowCount(0)
        self.correction_tab.y_ticks_table.setRowCount(0)
        for widget in self.series_widgets:
            widget.deleteLater()
        self.series_widgets = []
        self.results_tab.plot_label.setText("Recreated plot will appear here.")
        self.correction_tab.apply_corrections_btn.setEnabled(False)
        self.postprocessing_tab.apply_postprocessing_btn.setEnabled(False)
        self.postprocessing_tab.reset_series_btn.setEnabled(False)
        self.postprocessing_tab.reset_all_btn.setEnabled(False)
        if hasattr(self, 'save_tab'):
            self.save_tab.save_preview_table.setRowCount(0)
            self.save_tab.save_preview_table.setColumnCount(0)
        if hasattr(self, 'save_tab'):
            self.save_tab.export_csv_btn.setEnabled(False)
        if hasattr(self, 'save_tab'):
            self.save_tab.export_mat_btn.setEnabled(False)
        if hasattr(self, 'postprocessing_tab'):
            # Clear plot with default linear scales
            self.postprocessing_tab.postproc_canvas.update_plot([], [], x_scale='linear', y_scale='linear')
            self.postprocessing_tab.postproc_canvas.interactive_points = []
            self.postprocessing_tab.postproc_series_combo.clear()
        self.processed_results = None
        if clear_ocr_debug:
            for widget in self.ocr_debug_widgets:
                widget.deleteLater()
            self.ocr_debug_widgets = []
            if hasattr(self, 'rerun_all_ocr_btn'):
                self.rerun_all_ocr_btn.setEnabled(False)
        self.show_masks_action.setEnabled(False)
        self.show_masks_action.setChecked(False)
        self.mask_overlay_pixmap = None
        # --- NEW: Clear draggable ticks state ---
        self.correct_ticks_action.setEnabled(False)
        self.correct_ticks_action.setChecked(False)
        self.image_viewer.show_draggable_ticks(False)
        if clear_image_viewer_state:
            self.image_viewer.set_image(QPixmap())  # Clear image viewer
