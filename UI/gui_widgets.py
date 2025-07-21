# UI/gui_widgets.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGroupBox, QSizePolicy, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea,
                             QTabWidget, QCheckBox)
from PyQt5.QtGui import (QPainter, QPen, QColor, QBrush, QFont, QPainterPath, QImage,
                         QPixmap)
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class Magnifier(QLabel):
    """A circular magnifying glass widget with crosshairs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(151, 151)
        # --- We no longer use a mask. We will clip the drawing for better performance. ---
        self.source_pixmap = None
        self.source_rect = QRect()
        self.hide()

    def update_source(self, pixmap: 'QPixmap', rect: QRect):
        """
        Efficiently updates the source for the magnifier.
        This avoids creating new pixmaps on every mouse move.
        """
        self.source_pixmap = pixmap
        self.source_rect = rect
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        """
        --- High-performance paint event. ---
        This draws the magnified view directly without creating intermediate images.
        """
        if not self.source_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create a circular clipping path to make the magnifier round
        path = QPainterPath()
        # --- FIX: Convert the integer QRect to a floating-point QRectF ---
        path.addEllipse(QRectF(self.rect()))
        painter.setClipPath(path)

        # Draw the magnified portion of the source pixmap directly onto the widget
        painter.drawPixmap(self.rect(), self.source_pixmap, self.source_rect)

        # Draw crosshairs
        pen = QPen(QColor(255, 0, 0, 180), 1, Qt.SolidLine)
        painter.setPen(pen)
        center_x, center_y = self.width() // 2, self.height() // 2
        painter.drawLine(center_x, 0, center_x, self.height())
        painter.drawLine(0, center_y, self.width(), center_y)

        # Draw border
        pen.setColor(QColor(0, 0, 0, 220))
        pen.setWidth(2)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(1, 1, self.width() - 2, self.height() - 2)
        painter.end()


class SeriesEditorWidget(QGroupBox):
    """A widget card for editing a single data series."""

    ocrRequested = pyqtSignal()
    deleteStateChanged = pyqtSignal()
    nameChanged = pyqtSignal()
    duplicateRequested = pyqtSignal()

    def __init__(self, series_name, color, original_index, parent=None):
        super().__init__(parent)
        self.original_name = series_name
        self.original_index = original_index
        self.is_deleted = False

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        main_layout = QVBoxLayout(self)

        # Header with name and delete button
        header_layout = QHBoxLayout()
        self.color_label = QLabel()
        self.color_label.setFixedSize(16, 16)
        self.set_series_color(color)

        self.name_label = QLabel(f"<b>Series:</b>")
        self.name_edit = QLineEdit(series_name)
        # let's not change everything while editing the series name we can do this after clicking on apply corrections
        # self.name_edit.textChanged.connect(self.nameChanged.emit)

        style = self.style()

        self.ocr_btn = QPushButton()
        ocr_icon = style.standardIcon(style.SP_FileDialogDetailedView)
        self.ocr_btn.setIcon(ocr_icon)
        self.ocr_btn.setToolTip("Detect series name from legend using OCR")
        self.ocr_btn.setFixedWidth(30)
        self.ocr_btn.clicked.connect(self.request_ocr)

        self.duplicate_btn = QPushButton()
        # Using a "new folder" icon as a stand-in for "copy" or "duplicate"
        duplicate_icon = style.standardIcon(style.SP_FileDialogNewFolder)
        self.duplicate_btn.setIcon(duplicate_icon)
        self.duplicate_btn.setToolTip("Duplicate this series")
        self.duplicate_btn.setFixedWidth(30)
        self.duplicate_btn.clicked.connect(self.request_duplication)

        self.delete_icon = style.standardIcon(style.SP_TrashIcon)
        self.undelete_icon = style.standardIcon(style.SP_DialogResetButton)  # A "restore" icon

        self.delete_btn = QPushButton()
        self.delete_btn.setIcon(self.delete_icon)
        self.delete_btn.setToolTip("Delete this series")
        self.delete_btn.setFixedWidth(30)
        self.delete_btn.clicked.connect(self.toggle_deleted_state)

        header_layout.addWidget(self.color_label)
        header_layout.addWidget(self.name_label)
        header_layout.addWidget(self.name_edit)
        header_layout.addWidget(self.ocr_btn)
        header_layout.addWidget(self.duplicate_btn)
        header_layout.addWidget(self.delete_btn)
        main_layout.addLayout(header_layout)

    def set_series_color(self, color: QColor):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 15, 15)
        painter.end()
        self.color_label.setPixmap(pixmap)

    def get_series_name(self):
        return self.name_edit.text()

    def request_ocr(self):
        self.ocrRequested.emit()

    def request_duplication(self):
        self.duplicateRequested.emit()

    def toggle_deleted_state(self):
        """Toggles the deleted status of the series and updates the UI accordingly."""
        self.is_deleted = not self.is_deleted
        self.deleteStateChanged.emit()
        if self.is_deleted:
            self.setTitle(f"{self.get_series_name()} (Deleted)")
            self.setStyleSheet("background-color: #ffdddd;")
            self.delete_btn.setIcon(self.undelete_icon)
            self.delete_btn.setToolTip("Restore this series")
            self.name_edit.setEnabled(False)
            self.duplicate_btn.setEnabled(False)
        else:
            self.setTitle("")
            self.setStyleSheet("")  # Revert to default stylesheet
            self.delete_btn.setIcon(self.delete_icon)
            self.delete_btn.setToolTip("Delete this series")
            self.name_edit.setEnabled(True)
            self.duplicate_btn.setEnabled(True)

class OcrDebugWidget(QGroupBox):
    """
    A simplified widget card to display OCR debug info for a single text region.
    It no longer contains controls, only displays information.
    """

    def __init__(self, debug_info, parent=None):
        super().__init__(parent)
        self.debug_info = debug_info  # Store for reference
        self.setTitle(debug_info.get('label', 'OCR Debug Item'))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        main_layout = QVBoxLayout(self)

        # Image display
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        self.original_crop_label = QLabel("Original")
        self.processed_img_label = QLabel("Processed")
        self.original_crop_label.setAlignment(Qt.AlignCenter)
        self.processed_img_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.original_crop_label)
        image_layout.addWidget(self.processed_img_label)
        image_layout.addStretch()
        main_layout.addLayout(image_layout)

        # Result display
        result_layout = QFormLayout()
        self.predicted_text_edit = QLineEdit()
        self.predicted_text_edit.setReadOnly(True)
        result_layout.addRow("Predicted Text:", self.predicted_text_edit)
        main_layout.addLayout(result_layout)

        self.populate_initial_data()

    def _numpy_to_pixmap(self, np_img, height=60):
        if np_img is None or np_img.size == 0:
            return QPixmap()

        if len(np_img.shape) == 2:  # Grayscale
            h, w = np_img.shape
            q_img = QImage(np_img.data, w, h, w, QImage.Format_Grayscale8)
        else:  # BGR/BGRA
            np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            h, w, ch = np_img_rgb.shape
            q_img = QImage(np_img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        return QPixmap.fromImage(q_img).scaledToHeight(height, Qt.SmoothTransformation)

    def populate_initial_data(self):
        self.update_contents(
            self.debug_info.get('text', ''),
            self.debug_info.get('processed_img')
        )
        original_pixmap = self._numpy_to_pixmap(self.debug_info.get('original_crop'))
        self.original_crop_label.setPixmap(original_pixmap)

    def update_contents(self, new_text, new_processed_np_img):
        """Updates the widget with new OCR results."""
        self.predicted_text_edit.setText(new_text)
        pixmap = self._numpy_to_pixmap(new_processed_np_img)
        self.processed_img_label.setPixmap(pixmap)

        # Also update the title to reflect the new text
        original_label = self.debug_info.get('label', 'OCR Item: ...').split(':')[0]
        self.setTitle(f"{original_label}: {new_text[:15]}...")

class InteractivePlotCanvas(FigureCanvas):
    """A matplotlib canvas that supports interactive point editing."""
    points_updated = pyqtSignal(list)

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.interactive_points = []
        self.line = None
        self.interactive_scatter = None
        # --- NEW: Add state for dragging ---
        self.dragged_point_index = None

        # --- MODIFIED: Connect more events for dragging ---
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def update_plot(self, processed_series, colors, original_series=None, interactive_mode=False,
                    x_scale='linear', y_scale='linear'):
        self.axes.clear()

        # --- NEW: Set axis scales before plotting ---
        self.axes.set_xscale(x_scale)
        self.axes.set_yscale(y_scale)

        # The state of interactive_points is managed externally by the main app logic.
        # This method should only draw the current state, not modify it.
        self.line = None
        self.interactive_scatter = None
        self.dragged_point_index = None

        # Plot original data first if available, with a distinct style
        if original_series:
            for i, series in enumerate(original_series):
                points = series['data_points']
                if not points: continue

                original_index = series.get('original_index', i)
                color = colors[original_index % len(colors)]
                # Make original line semi-transparent
                q_color = QColor(color.name())
                q_color.setAlpha(180)

                x = [p['x'] for p in points]
                y = [p['y'] for p in points]
                self.axes.plot(x, y, color=q_color.name(QColor.HexRgb), linestyle='--', alpha=0.7,
                               label=f"{series['series_name']} (Original)")

        # Plot the main (processed) data
        if not processed_series:
            if not original_series:  # Only show placeholder text if the canvas is completely empty
                self.axes.text(0.5, 0.5, "Apply processing or select a series to view data.", ha='center', va='center')
            self.axes.legend()
            self.draw()
            return

        for i, series in enumerate(processed_series):
            points = series['data_points']
            if not points: continue

            original_index = series.get('original_index', i)
            color = colors[original_index % len(colors)]

            x_line = [p['x'] for p in points]
            y_line = [p['y'] for p in points]

            # Handle interactive mode for a single series
            if interactive_mode and len(processed_series) == 1:
                # The line is the interpolated data from processed_series
                self.line, = self.axes.plot(x_line, y_line, color=color.name(QColor.HexRgb), label=series['series_name'])

                # The scatter points are the user-editable reference points, which are stored on the canvas itself.
                # We do NOT overwrite self.interactive_points here.
                if self.interactive_points:
                    x_scatter = [p['x'] for p in self.interactive_points]
                    y_scatter = [p['y'] for p in self.interactive_points]
                    self.interactive_scatter = self.axes.scatter(x_scatter, y_scatter, color='red', s=50, zorder=5, label='Interactive Points')
            else:
                self.axes.plot(x_line, y_line, color=color.name(QColor.HexRgb), label=series['series_name'])

        # Final plot styling
        self.axes.minorticks_on()
        # Use which='both' to handle grid lines correctly for linear and log scales
        self.axes.grid(True, which='both', linestyle='--', linewidth='0.5')
        if self.axes.get_legend_handles_labels()[1]:
            self.axes.legend()
        self.fig.tight_layout()
        self.draw()

    def on_press(self, event):
        """Selects a point when the user clicks on it."""
        # Only handle left-clicks inside the plot area when in interactive mode
        if not self.interactive_points or event.inaxes != self.axes or event.button != 1:
            return

        # Find the closest point to the click
        x_data, y_data = self.interactive_scatter.get_offsets().T
        click_display_coords = np.array([event.x, event.y])
        points_display_coords = self.axes.transData.transform(np.vstack([x_data, y_data]).T)
        distances = np.sqrt(np.sum((points_display_coords - click_display_coords) ** 2, axis=1))

        if len(distances) == 0:
            return

        closest_index = np.argmin(distances)
        # Set a threshold for picking up a point (e.g., 10 pixels)
        if distances[closest_index] < 10:
            self.dragged_point_index = closest_index

    def on_motion(self, event):
        """Handles dragging a selected point."""
        if self.dragged_point_index is None or event.inaxes != self.axes:
            return

        # Update the point's position
        self.interactive_points[self.dragged_point_index]['x'] = event.xdata
        self.interactive_points[self.dragged_point_index]['y'] = event.ydata

        # Update the visual representation of the points
        if self.interactive_scatter:
            self.interactive_scatter.set_offsets(np.c_[[p['x'] for p in self.interactive_points], [p['y'] for p in self.interactive_points]])
            self.draw_idle()

    def on_release(self, event):
        """Stops dragging when the mouse button is released."""
        if event.button == 1 and self.dragged_point_index is not None:
            self.dragged_point_index = None
            self.points_updated.emit(self.interactive_points)

class CorrectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        ocr_group = QGroupBox("OCR & Axis Corrections")
        ocr_layout = QFormLayout(ocr_group)
        style = self.style()
        ocr_icon = style.standardIcon(style.SP_FileDialogDetailedView)

        plot_title_layout = QHBoxLayout()
        self.plot_title_edit = QLineEdit()
        self.plot_title_ocr_btn = QPushButton()
        self.plot_title_ocr_btn.setIcon(ocr_icon)
        plot_title_layout.addWidget(self.plot_title_edit)
        plot_title_layout.addWidget(self.plot_title_ocr_btn)
        ocr_layout.addRow("Plot Title:", plot_title_layout)

        x_title_layout = QHBoxLayout()
        self.x_title_edit = QLineEdit()
        self.x_title_ocr_btn = QPushButton()
        self.x_title_ocr_btn.setIcon(ocr_icon)
        x_title_layout.addWidget(self.x_title_edit)
        x_title_layout.addWidget(self.x_title_ocr_btn)
        ocr_layout.addRow("X-Axis Title:", x_title_layout)

        y_title_layout = QHBoxLayout()
        self.y_title_edit = QLineEdit()
        self.y_title_ocr_btn = QPushButton()
        self.y_title_ocr_btn.setIcon(ocr_icon)
        y_title_layout.addWidget(self.y_title_edit)
        y_title_layout.addWidget(self.y_title_ocr_btn)
        ocr_layout.addRow("Y-Axis Title:", y_title_layout)

        ocr_layout.addRow(QLabel("<b>X-Axis Ticks</b>"))
        self.x_ticks_table = QTableWidget()
        self.x_ticks_table.setColumnCount(3)
        self.x_ticks_table.setHorizontalHeaderLabels(["Tick Image", "Predicted Text", "Corrected Value"])
        self.x_ticks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.x_ticks_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.x_ticks_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.x_ticks_table.setMaximumHeight(150)
        ocr_layout.addRow(self.x_ticks_table)

        ocr_layout.addRow(QLabel("<b>Y-Axis Ticks</b>"))
        self.y_ticks_table = QTableWidget()
        self.y_ticks_table.setColumnCount(3)
        self.y_ticks_table.setHorizontalHeaderLabels(["Tick Image", "Predicted Text", "Corrected Value"])
        self.y_ticks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.y_ticks_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.y_ticks_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.y_ticks_table.setMaximumHeight(150)
        ocr_layout.addRow(self.y_ticks_table)

        series_group = QGroupBox("Data Series")
        series_layout = QVBoxLayout(series_group)
        self.series_scroll_area = QScrollArea()
        self.series_scroll_area.setWidgetResizable(True)
        self.series_widget_container = QWidget()
        self.series_container_layout = QVBoxLayout(self.series_widget_container)
        self.series_scroll_area.setWidget(self.series_widget_container)
        series_layout.addWidget(self.series_scroll_area)

        interpolation_group = QGroupBox("Recalculation Options")
        interpolation_layout = QFormLayout(interpolation_group)
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Linear", "Cubic Spline", "None (Raw Points)"])
        interpolation_layout.addRow("Interpolation Method:", self.interpolation_combo)

        self.apply_corrections_btn = QPushButton(" Apply All Corrections & Update Plot")
        self.apply_corrections_btn.setIcon(self.style().standardIcon(self.style().SP_DialogApplyButton))
        self.apply_corrections_btn.setEnabled(False)

        layout.addWidget(ocr_group)
        layout.addWidget(series_group)
        layout.addWidget(interpolation_group)
        layout.addWidget(self.apply_corrections_btn)

class PostProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        controls_container = QWidget()
        controls_container.setMaximumWidth(400)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setAlignment(Qt.AlignTop)

        series_selection_layout = QFormLayout()
        self.postproc_series_combo = QComboBox()
        series_selection_layout.addRow("Target Series:", self.postproc_series_combo)
        controls_layout.addLayout(series_selection_layout)

        self.postprocessing_tabs = QTabWidget()

        filter_tab = QWidget()
        filter_layout = QFormLayout(filter_tab)
        filter_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)

        self.outlier_group = QGroupBox("Outlier Removal (Local Regression)")
        self.outlier_group.setToolTip("Removes data points that deviate significantly from their neighbors. Disabled during manual editing.")
        outlier_form_layout = QFormLayout(self.outlier_group)
        
        self.outlier_enabled_check = QCheckBox("Enable")

        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["Local Regression", "RANSAC"])
        self.outlier_method_combo.setToolTip("Local Regression is good for general noise. RANSAC is better for charts with significant, large outliers.")

        self.outlier_window_spin = QDoubleSpinBox()
        self.outlier_window_spin.setRange(0.05, 0.5)
        self.outlier_window_spin.setValue(0.15)
        self.outlier_window_spin.setSingleStep(0.01)
        self.outlier_window_spin.setDecimals(2)
        self.outlier_window_label = QLabel("Window Fraction:") # Label to toggle

        self.outlier_threshold_spin = QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(1.0, 10.0)
        self.outlier_threshold_spin.setValue(3.0)
        self.outlier_threshold_spin.setSingleStep(0.1)
        outlier_form_layout.addRow(self.outlier_enabled_check)
        outlier_form_layout.addRow("Method:", self.outlier_method_combo)
        outlier_form_layout.addRow(self.outlier_window_label, self.outlier_window_spin)
        outlier_form_layout.addRow("Threshold (Std Dev):", self.outlier_threshold_spin)
        filter_layout.addRow(self.outlier_group)

        # --- Auto Resampling Group ---
        auto_resampling_group = QGroupBox("Auto Resampling")
        auto_resampling_group.setToolTip("Automatically redraws the series with a specified number of points.")
        auto_resampling_form_layout = QFormLayout(auto_resampling_group)
        self.auto_resampling_enabled_check = QCheckBox("Enable")
        self.auto_resample_points_spin = QSpinBox()
        self.auto_resample_points_spin.setRange(3, 1000)
        self.auto_resample_points_spin.setValue(100)
        self.auto_resample_method_combo = QComboBox()
        self.auto_resample_method_combo.addItems(["Linear", "Cubic Spline", "Savitzky-Golay"])
        auto_resampling_form_layout.addRow(self.auto_resampling_enabled_check)
        auto_resampling_form_layout.addRow("Number of Points:", self.auto_resample_points_spin)
        auto_resampling_form_layout.addRow("Method:", self.auto_resample_method_combo)
        filter_layout.addRow(auto_resampling_group)

        # --- Manual Editing Group ---
        manual_editing_group = QGroupBox("Manual Editing & Interpolation")
        manual_editing_group.setToolTip("Manually place reference points and then interpolate between them.")
        manual_editing_form_layout = QFormLayout(manual_editing_group)
        self.manual_editing_enabled_check = QCheckBox("Enable")
        self.manual_ref_points_spin = QSpinBox()
        self.manual_ref_points_spin.setRange(3, 100)
        self.manual_ref_points_spin.setValue(15)
        self.manual_init_reset_btn = QPushButton("Initialize/Reset Reference Points")
        self.manual_final_points_spin = QSpinBox()
        self.manual_final_points_spin.setRange(10, 2000)
        self.manual_final_points_spin.setValue(100)
        self.manual_interp_method_combo = QComboBox()
        self.manual_interp_method_combo.addItems(["Linear", "Cubic Spline", "Savitzky-Golay"])
        manual_editing_form_layout.addRow(self.manual_editing_enabled_check)
        manual_editing_form_layout.addRow("Num. Reference Points:", self.manual_ref_points_spin)
        manual_editing_form_layout.addRow(self.manual_init_reset_btn)
        manual_editing_form_layout.addRow("Final Num. Points (Resolution):", self.manual_final_points_spin)
        manual_editing_form_layout.addRow("Interpolation Method:", self.manual_interp_method_combo)
        filter_layout.addRow(manual_editing_group)

        self.postprocessing_tabs.addTab(filter_tab, "Filtering & Smoothing")

        controls_layout.addWidget(self.postprocessing_tabs)

        view_options_group = QGroupBox("View Options")
        view_options_layout = QHBoxLayout(view_options_group)
        self.show_original_check = QCheckBox("Show Original Data Overlay")
        self.show_original_check.setToolTip("Toggles a view of the original data before post-processing.")
        view_options_layout.addWidget(self.show_original_check)
        controls_layout.addWidget(view_options_group)

        self.apply_postprocessing_btn = QPushButton("Apply Post-processing")
        self.apply_postprocessing_btn.setEnabled(False)

        self.reset_series_btn = QPushButton("Reset Selected Series")
        self.reset_series_btn.setEnabled(False)

        self.reset_all_btn = QPushButton("Reset All")
        self.reset_all_btn.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_postprocessing_btn)
        btn_layout.addWidget(self.reset_series_btn)
        btn_layout.addWidget(self.reset_all_btn)
        controls_layout.addLayout(btn_layout)

        self.postproc_canvas = InteractivePlotCanvas(self)

        plot_toolbar = NavigationToolbar(self.postproc_canvas, self)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.addWidget(plot_toolbar)
        plot_layout.addWidget(self.postproc_canvas)

        layout.addWidget(controls_container)
        layout.addWidget(plot_container, 1)

class SaveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        preview_group = QGroupBox("Final Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.save_preview_table = QTableWidget()
        self.save_preview_table.setColumnCount(0)
        self.save_preview_table.setAlternatingRowColors(True)
        self.save_preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        preview_layout.addWidget(self.save_preview_table)
        layout.addWidget(preview_group)

        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout(export_group)
        self.export_csv_btn = QPushButton("Export as CSV")
        self.export_mat_btn = QPushButton("Export as .mat")
        self.export_csv_btn.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))
        self.export_mat_btn.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_mat_btn)
        layout.addWidget(export_group)

        self.export_csv_btn.setEnabled(False)
        self.export_mat_btn.setEnabled(False)

class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plot_label = QLabel("Recreated plot will appear here.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)
