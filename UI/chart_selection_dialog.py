# UI/chart_selection_dialog.py

from PyQt5.QtWidgets import (QDialog, QScrollArea, QWidget, QGridLayout,
                             QPushButton, QDialogButtonBox, QVBoxLayout, QLabel)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QSize, Qt
import cv2

class ChartSelectionDialog(QDialog):
    """
    A dialog window to display extracted charts in a grid and allow the user to select one.
    """
    chartSelected = pyqtSignal(QPixmap)

    def __init__(self, chart_images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a Chart to Import")
        self.setMinimumSize(800, 600)
        self.chart_pixmaps = []

        main_layout = QVBoxLayout(self)

        if not chart_images:
            info_label = QLabel("No charts were detected in the selected PDF.")
            info_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(info_label)
        else:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            
            container = QWidget()
            grid_layout = QGridLayout(container)
            grid_layout.setSpacing(15)

            # Convert numpy images to QPixmap and store them
            for img_np in chart_images:
                self.chart_pixmaps.append(self._numpy_to_pixmap(img_np))

            # Populate the grid
            cols = 3 # Number of columns in the grid
            for i, pixmap in enumerate(self.chart_pixmaps):
                row, col = divmod(i, cols)
                
                btn = QPushButton()
                btn.setIcon(QIcon(pixmap))
                btn.setIconSize(QSize(250, 200))
                btn.setFixedSize(260, 210)
                btn.clicked.connect(self.create_selection_handler(i))
                grid_layout.addWidget(btn, row, col)

            scroll_area.setWidget(container)
            main_layout.addWidget(scroll_area)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _numpy_to_pixmap(self, np_img):
        """Converts a BGR NumPy image to a QPixmap."""
        if np_img is None or np_img.size == 0: return QPixmap()
        np_img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        h, w, ch = np_img_rgb.shape
        q_img = QImage(np_img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def create_selection_handler(self, index):
        """Creates a closure to handle button clicks and emit the correct pixmap."""
        def handler():
            self.chartSelected.emit(self.chart_pixmaps[index])
            self.accept()
        return handler