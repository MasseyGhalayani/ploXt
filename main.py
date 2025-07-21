# main_gui.py
import sys

from PyQt5.QtWidgets import QApplication

# --- NEW: Import the splash screen and main window ---
from UI.splash_screen import SplashScreen
from UI.app import MainAppWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 1. Create and show the splash screen
    splash = SplashScreen()
    splash.show()

    # Process events to make sure the splash screen is drawn before heavy lifting
    app.processEvents()

    # 2. Create the main window (it's not shown yet)
    window = MainAppWindow()

    # 3. When model loading is complete, close the splash and show the main window
    window.initialization_complete.connect(splash.close)
    window.initialization_complete.connect(window.show)

    # 4. Start the slow model initialization process in a background thread
    window.start_model_initialization(splash)

    sys.exit(app.exec_())