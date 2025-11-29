"""
Main window for Hoggle application.
"""
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from core.spell_engine import SpellType
from core.voice_verifier import VoiceVerifier
from gui.camera_widget import CameraWidget


class SpellVerificationThread(QThread):
    """Thread for spell verification to avoid blocking UI."""
    
    verification_complete = pyqtSignal(bool, str)
    
    def __init__(self, voice_verifier: VoiceVerifier, target_spell: str):
        super().__init__()
        self.voice_verifier = voice_verifier
        self.target_spell = target_spell
    
    def run(self):
        """Run verification."""
        try:
            is_correct, feedback = self.voice_verifier.verify_and_feedback(self.target_spell)
            self.verification_complete.emit(is_correct, feedback)
        except Exception as e:
            self.verification_complete.emit(False, f"Error: {str(e)}")


class SpellSelectionScreen(QWidget):
    """Initial screen for selecting a spell to learn."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("Hoggle - Learn Spells with Hermione!")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Choose a spell to practice:")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        # Spell buttons
        spells = [
            ("Lumos", "Light up your wand tip", SpellType.LUMOS),
            ("Accio", "Bring objects to you", SpellType.ACCIO),
            ("Wingardium Leviosa", "Make objects fly", SpellType.WINGARDIUM_LEVIOSA)
        ]
        
        for spell_name, description, spell_type in spells:
            btn = QPushButton(f"{spell_name}\n{description}")
            btn.setFont(QFont("Arial", 12))
            btn.setMinimumHeight(80)
            btn.clicked.connect(lambda checked, st=spell_type: self.select_spell(st))
            layout.addWidget(btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_spell(self, spell_type: SpellType):
        """Select a spell to practice."""
        if self.parent_window:
            self.parent_window.show_practice_screen(spell_type)


class PracticeScreen(QWidget):
    """Screen for practicing a spell."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_spell: SpellType = None
        self.voice_verifier: VoiceVerifier = None
        self.verification_thread: SpellVerificationThread = None
        self.practice_started = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with back button
        header_layout = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self.go_back)
        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        
        self.spell_title = QLabel("")
        self.spell_title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(self.spell_title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel(""))  # Spacer for balance
        
        layout.addLayout(header_layout)
        
        # Spell description
        self.description = QTextEdit()
        self.description.setReadOnly(True)
        self.description.setMaximumHeight(100)
        self.description.setFont(QFont("Arial", 11))
        layout.addWidget(self.description)
        
        # Camera widget
        self.camera_widget = CameraWidget()
        layout.addWidget(self.camera_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_practice_btn = QPushButton("Start Practice")
        self.start_practice_btn.setFont(QFont("Arial", 12))
        self.start_practice_btn.clicked.connect(self.start_practice)
        button_layout.addWidget(self.start_practice_btn)
        
        self.cast_spell_btn = QPushButton("Cast Spell (Hold to Record)")
        self.cast_spell_btn.setFont(QFont("Arial", 12))
        self.cast_spell_btn.setEnabled(False)
        self.cast_spell_btn.pressed.connect(self.start_recording)
        self.cast_spell_btn.released.connect(self.stop_recording)
        button_layout.addWidget(self.cast_spell_btn)
        
        layout.addLayout(button_layout)
        
        # Feedback label
        self.feedback_label = QLabel("")
        self.feedback_label.setFont(QFont("Arial", 11))
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.feedback_label)
        
        self.setLayout(layout)
    
    def set_spell(self, spell_type: SpellType):
        """Set the current spell to practice."""
        self.current_spell = spell_type
        
        spell_info = {
            SpellType.LUMOS: {
                "name": "Lumos",
                "description": "Lumos creates light at the tip of your wand. Say 'Lumos' clearly and point your wand forward."
            },
            SpellType.ACCIO: {
                "name": "Accio",
                "description": "Accio summons objects to you. Say 'Accio' and point your wand at the object you want to summon."
            },
            SpellType.WINGARDIUM_LEVIOSA: {
                "name": "Wingardium Leviosa",
                "description": "Wingardium Leviosa makes objects float. Say 'Wingardium Leviosa' (it's Levi-O-sa, not Levio-SA!) and wave your wand."
            }
        }
        
        info = spell_info[spell_type]
        self.spell_title.setText(info["name"])
        self.description.setText(info["description"])
        self.feedback_label.setText("")
        self.practice_started = False
        self.start_practice_btn.setEnabled(True)
        self.cast_spell_btn.setEnabled(False)
    
    def start_practice(self):
        """Start practice mode."""
        try:
            # Initialize voice verifier
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                QMessageBox.warning(
                    self,
                    "API Key Missing",
                    "Please set GOOGLE_API_KEY environment variable."
                )
                return
            
            self.voice_verifier = VoiceVerifier(api_key=api_key)
            
            # Start camera
            self.camera_widget.start_camera()
            
            self.practice_started = True
            self.start_practice_btn.setEnabled(False)
            self.cast_spell_btn.setEnabled(True)
            self.feedback_label.setText("Practice mode started! Hold the 'Cast Spell' button and speak the spell.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start practice: {str(e)}")
    
    def start_recording(self):
        """Start recording audio."""
        if not self.practice_started or not self.voice_verifier:
            return
        
        self.feedback_label.setText("Recording... Speak the spell now!")
        self.cast_spell_btn.setEnabled(False)
    
    def stop_recording(self):
        """Stop recording and verify spell."""
        if not self.practice_started or not self.voice_verifier:
            return
        
        self.feedback_label.setText("Verifying with Hermione...")
        
        # Get spell name
        spell_names = {
            SpellType.LUMOS: "Lumos",
            SpellType.ACCIO: "Accio",
            SpellType.WINGARDIUM_LEVIOSA: "Wingardium Leviosa"
        }
        target_spell = spell_names[self.current_spell]
        
        # Run verification in thread
        self.verification_thread = SpellVerificationThread(self.voice_verifier, target_spell)
        self.verification_thread.verification_complete.connect(self.on_verification_complete)
        self.verification_thread.start()
    
    def on_verification_complete(self, is_correct: bool, feedback: str):
        """Handle verification result."""
        self.cast_spell_btn.setEnabled(True)
        self.feedback_label.setText(feedback)
        
        if is_correct:
            # Activate spell effect
            spell_engine = self.camera_widget.get_spell_engine()
            # Activate spell - wand position will be updated in the camera thread
            # Pass None to use center as default, update loop will position it correctly
            spell_engine.activate_spell(self.current_spell, None)
            self.feedback_label.setStyleSheet(
                "padding: 10px; background-color: #d4edda; border-radius: 5px; color: #155724;"
            )
        else:
            self.feedback_label.setStyleSheet(
                "padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #721c24;"
            )
    
    def go_back(self):
        """Return to spell selection."""
        if self.camera_widget:
            self.camera_widget.stop_camera()
        if self.voice_verifier:
            self.voice_verifier.release()
        if self.parent_window:
            self.parent_window.show_spell_selection()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hoggle - Learn Spells with Hermione")
        self.setMinimumSize(800, 600)
        
        # Create stacked widget for screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create screens
        self.spell_selection_screen = SpellSelectionScreen(self)
        self.practice_screen = PracticeScreen(self)
        
        # Add screens to stack
        self.stacked_widget.addWidget(self.spell_selection_screen)
        self.stacked_widget.addWidget(self.practice_screen)
        
        # Show initial screen
        self.show_spell_selection()
    
    def show_spell_selection(self):
        """Show spell selection screen."""
        self.stacked_widget.setCurrentWidget(self.spell_selection_screen)
    
    def show_practice_screen(self, spell_type: SpellType):
        """Show practice screen for a spell."""
        self.practice_screen.set_spell(spell_type)
        self.stacked_widget.setCurrentWidget(self.practice_screen)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.practice_screen:
            self.practice_screen.camera_widget.stop_camera()
            if self.practice_screen.voice_verifier:
                self.practice_screen.voice_verifier.release()
        super().closeEvent(event)

