# main_gui.py
import sys

from PyQt5.QtWidgets import (QApplication)

from UI.app import MainAppWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainAppWindow()
    window.show()
    sys.exit(app.exec_())