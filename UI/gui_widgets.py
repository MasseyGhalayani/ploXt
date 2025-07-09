# UI/gui_widgets.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGroupBox, QSizePolicy)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF


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

    def __init__(self, series_name, dataframe, parent=None):
        super().__init__(parent)
        self.series_name = series_name
        self.is_deleted = False

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        main_layout = QVBoxLayout(self)

        # Header with name and delete button
        header_layout = QHBoxLayout()
        self.name_label = QLabel(f"<b>Series:</b>")
        self.name_edit = QLineEdit(series_name)

        style = self.style()
        delete_icon = style.standardIcon(style.SP_TrashIcon)
        self.delete_btn = QPushButton()
        self.delete_btn.setIcon(delete_icon)
        self.delete_btn.setToolTip("Delete this series")
        self.delete_btn.setFixedWidth(30)
        self.delete_btn.clicked.connect(self.mark_as_deleted)

        header_layout.addWidget(self.name_label)
        header_layout.addWidget(self.name_edit)
        header_layout.addWidget(self.delete_btn)
        main_layout.addLayout(header_layout)

        # Data table
        self.table = QTableWidget()
        self.table.setRowCount(dataframe.shape[0])
        self.table.setColumnCount(dataframe.shape[1])
        self.table.setHorizontalHeaderLabels(dataframe.columns)
        for row_idx, row in enumerate(dataframe.values):
            for col_idx, item in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))
        self.table.setFixedHeight(150)  # Limit height to make it scrollable
        main_layout.addWidget(self.table)

    def get_series_name(self):
        return self.name_edit.text()

    def mark_as_deleted(self):
        self.is_deleted = True
        self.setTitle(f"{self.series_name} (Deleted)")
        self.setStyleSheet("background-color: #ffdddd;")
        self.delete_btn.setEnabled(False)
        self.name_edit.setEnabled(False)