# UI/interactive_image_viewer.py
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal, QRectF

from UI.gui_widgets import Magnifier


class InteractiveImageViewer(QLabel):
    regionSelected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignCenter)

        # Image and Drawing State
        self.original_pixmap = None
        self.editable_pixmap = None
        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(255, 255, 255)

        # Zoom and Adaptive Brush
        self.zoom_levels = [60, 50, 40, 30, 20, 15, 10]
        self.brush_sizes = [20, 15, 10, 7, 5, 3, 2]
        self.current_zoom_index = 3

        # Tool State
        self.current_tool = 'brush'
        self.selection_start_point = None
        self.selection_rect = QRect()

        # Undo/Redo Stack
        self.undo_stack = []
        self.redo_stack = []

        # Child Widgets and Overlays
        self.magnifier = Magnifier(self)
        self.mask_overlay_pixmap = None
        self.masks_visible = False
        self.setMouseTracking(True)

    def set_image(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        # Handle null pixmap case
        if pixmap and not pixmap.isNull():
            self.editable_pixmap = self.original_pixmap.copy()
        else:
            self.editable_pixmap = QPixmap()

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.push_to_undo_stack()
        self.update_display()

    def set_tool(self, tool: str):
        self.current_tool = tool
        self.unsetCursor()

    def push_to_undo_stack(self):
        if self.editable_pixmap:
            self.undo_stack.append(self.editable_pixmap.copy())
            self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.editable_pixmap = self.undo_stack[-1].copy()
            self.update_display()

    def redo(self):
        if self.redo_stack:
            self.editable_pixmap = self.redo_stack.pop()
            self.undo_stack.append(self.editable_pixmap.copy())
            self.update_display()

    def get_adaptive_brush_size(self):
        """Returns the brush size corresponding to the current zoom level."""
        return self.brush_sizes[self.current_zoom_index]

    def get_edited_image_as_numpy(self):
        # --- FIX: Add robust checks for null pixmap/image ---
        if self.editable_pixmap is None or self.editable_pixmap.isNull():
            return None

        qimage = self.editable_pixmap.toImage().convertToFormat(4)
        if qimage.isNull():
            return None

        ptr = qimage.bits()
        if ptr is None:
            return None
        # --- End of Fix ---

        ptr.setsize(qimage.height() * qimage.width() * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((qimage.height(), qimage.width(), 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def wheelEvent(self, event):
        """Handle mouse wheel scrolling to change zoom level."""
        if self.editable_pixmap is None:
            return

        if not (event.buttons() & Qt.RightButton):
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self.current_zoom_index = max(0, self.current_zoom_index - 1)
        elif delta < 0:
            self.current_zoom_index = min(len(self.zoom_levels) - 1, self.current_zoom_index + 1)

        self.update_magnifier(event.pos())
        event.accept()

    def mousePressEvent(self, event):
        if self.editable_pixmap is None:
            return

        if event.button() == Qt.LeftButton:
            if self.current_tool == 'brush':
                self.drawing = True
                self.last_point = self.map_pos_to_pixmap(event.pos())
            elif self.current_tool in ['select_and_fill', 'select_title']:
                self.drawing = True
                self.selection_start_point = self.map_pos_to_pixmap(event.pos())
                self.selection_rect = QRect(self.selection_start_point, self.selection_start_point)

        elif event.button() == Qt.RightButton:
            self.update_magnifier(event.pos())

    def mouseMoveEvent(self, event):
        if self.editable_pixmap is None:
            return

        if event.buttons() & Qt.RightButton:
            self.update_magnifier(event.pos())
        else:
            self.magnifier.hide()

        if event.buttons() & Qt.LeftButton and self.drawing:
            current_point = self.map_pos_to_pixmap(event.pos())
            if self.current_tool == 'brush':
                painter = QPainter(self.editable_pixmap)
                brush_size = self.get_adaptive_brush_size()
                pen = QPen(self.brush_color, brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_point, current_point)
                painter.end()
                self.last_point = current_point
                self.update_display()
            elif self.current_tool in ['select_and_fill', 'select_title']:
                self.selection_rect = QRect(self.selection_start_point, current_point).normalized()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing:
                self.drawing = False
                if self.current_tool == 'select_and_fill':
                    painter = QPainter(self.editable_pixmap)
                    painter.fillRect(self.selection_rect, self.brush_color)
                    painter.end()
                    self.selection_rect = QRect()
                    self.update_display()
                    self.push_to_undo_stack()
                elif self.current_tool == 'brush':
                    self.push_to_undo_stack()
                elif self.current_tool == 'select_title':
                    bbox = list(self.selection_rect.getCoords())
                    self.regionSelected.emit(bbox)
                    self.selection_rect = QRect()
                    self.update()

        elif event.button() == Qt.RightButton:
            self.magnifier.hide()

    def paintEvent(self, event):
        """
        Overridden paint event to draw overlays on top of the base image.
        """
        super().paintEvent(event)  # Draws the base image (self.pixmap())

        # Use a single painter for all subsequent overlay drawings
        painter = QPainter(self)

        # Draw mask overlay if it's visible and available
        if self.masks_visible and self.mask_overlay_pixmap:
            offset_x, offset_y, scale_factor = self.get_scaled_pixmap_geometry()
            if scale_factor > 0:
                target_w = self.original_pixmap.width() * scale_factor
                target_h = self.original_pixmap.height() * scale_factor
                target_rect = QRectF(offset_x, offset_y, target_w, target_h)
                source_rect = QRectF(self.mask_overlay_pixmap.rect())
                painter.drawPixmap(target_rect, self.mask_overlay_pixmap, source_rect)

        # Draw selection rectangle if currently drawing
        if self.drawing and self.current_tool in ['select_and_fill', 'select_title'] and not self.selection_rect.isNull():
            pen = QPen(QColor(255, 0, 0, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            widget_rect = self.map_rect_to_widget(self.selection_rect)
            painter.drawRect(widget_rect)

    def enterEvent(self, event):
        pass

    def leaveEvent(self, event):
        self.magnifier.hide()

    def update_magnifier(self, pos):
        if self.editable_pixmap is None: return

        if not self.magnifier.isVisible():
            self.magnifier.show()

        self.magnifier.move(pos.x() - self.magnifier.width() // 2, pos.y() - self.magnifier.height() // 2)
        pixmap_pos = self.map_pos_to_pixmap(pos)
        source_rect_size = self.zoom_levels[self.current_zoom_index]
        source_rect = QRect(
            pixmap_pos.x() - source_rect_size // 2,
            pixmap_pos.y() - source_rect_size // 2,
            source_rect_size, source_rect_size
        )
        self.magnifier.update_source(self.editable_pixmap, source_rect)

    def resizeEvent(self, event):
        if self.editable_pixmap:
            self.update_display()

    def update_display(self):
        if self.editable_pixmap and not self.editable_pixmap.isNull():
            self.setPixmap(self.editable_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setPixmap(QPixmap()) # Clear the label if pixmap is null
        self.update() # Ensure overlays are redrawn

    def get_scaled_pixmap_geometry(self):
        if self.original_pixmap is None or self.original_pixmap.isNull(): return 0, 0, 0, 0
        original_w, original_h = self.original_pixmap.width(), self.original_pixmap.height()
        widget_w, widget_h = self.width(), self.height()
        if original_w == 0 or original_h == 0: return 0, 0, 0, 0
        scale_factor = min(widget_w / original_w, widget_h / original_h)
        scaled_w, scaled_h = original_w * scale_factor, original_h * scale_factor
        offset_x, offset_y = (widget_w - scaled_w) / 2, (widget_h - scaled_h) / 2
        return offset_x, offset_y, scale_factor

    def map_pos_to_pixmap(self, pos: QPoint) -> QPoint:
        offset_x, offset_y, scale_factor = self.get_scaled_pixmap_geometry()
        if scale_factor == 0: return pos
        pixmap_x = (pos.x() - offset_x) / scale_factor
        pixmap_y = (pos.y() - offset_y) / scale_factor
        return QPoint(int(pixmap_x), int(pixmap_y))

    def map_rect_to_widget(self, rect: QRect) -> QRect:
        offset_x, offset_y, scale_factor = self.get_scaled_pixmap_geometry()
        if scale_factor == 0: return rect
        widget_x = rect.x() * scale_factor + offset_x
        widget_y = rect.y() * scale_factor + offset_y
        widget_w = rect.width() * scale_factor
        widget_h = rect.height() * scale_factor
        return QRect(int(widget_x), int(widget_y), int(widget_w), int(widget_h))

    def set_mask_overlay(self, overlay_pixmap):
        """Stores the pixmap to be used for the mask overlay."""
        self.mask_overlay_pixmap = overlay_pixmap
        self.update()

    def show_masks(self, show):
        """Sets the visibility of the mask overlay."""
        self.masks_visible = show
        self.update()