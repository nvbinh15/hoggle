"""
Main entry point for Hoggle application.
"""
import sys
import os
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Run the Hoggle application."""
    app = QApplication(sys.argv)
    
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Please set it before running the application:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

