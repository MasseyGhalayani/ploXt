# UI/splash_screen.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        # --- Window Flags for a frameless, splash-like window ---
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(450, 250)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Title Label ---
        self.title_label = QLabel("PloXt")
        font = self.title_label.font()
        font.setPointSize(48)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #333;")

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #777; border-radius: 5px; background-color: #E0E0E0; height: 15px; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 4px; }
        """)

        # --- Status Message Label ---
        self.message_label = QLabel("Initializing...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #555; font-size: 10pt;")

        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.message_label)
        layout.addWidget(self.progress_bar)

    def set_progress(self, value):
        self.progress_bar.setValue(value)

    def set_message(self, message):
        self.message_label.setText(message)

    def paintEvent(self, event):
        """Custom paint event to draw a rounded background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(245, 245, 245, 250))
        painter.setPen(QColor(150, 150, 150))
        painter.drawRoundedRect(self.rect(), 10, 10)