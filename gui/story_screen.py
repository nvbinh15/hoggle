import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from core.voice_verifier import VoiceVerifier
from core.story_manager import StoryManager, StoryStepType
from gui.camera_widget import CameraWidget
from gui.utils import SpellVerificationThread

class StoryScreen(QWidget):
    """Game screen handling the storyline and spell casting."""
    
    finished = pyqtSignal()  # Signal when game is completed or exited

    def __init__(self, parent=None):
        super().__init__(parent)
        self.story_manager = StoryManager()
        self.voice_verifier: VoiceVerifier = None
        self.verification_thread: SpellVerificationThread = None
        self.is_listening = False
        self.level_completed = False
        self.game_active = False
        self.auto_retry_timer = QTimer(self)
        self.auto_retry_timer.setSingleShot(True)
        self.auto_retry_timer.timeout.connect(self.start_listening)

        self.success_timer = QTimer(self)
        self.success_timer.setSingleShot(True)
        self.success_timer.timeout.connect(self.next_level)
        
        self.init_ui()
        
    def set_status_message(self, text: str, style_type: str = "info"):
        """
        Update status label with consistent styling.
        style_type: 'info', 'listening', 'success', 'error'
        """
        base_style = """
            QLabel {
                padding: 15px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid;
            }
        """
        
        styles = {
            "info": """
                background-color: #2c3e50;
                color: #ecf0f1;
                border-color: #34495e;
            """,
            "listening": """
                background-color: #f39c12;
                color: #ffffff;
                border-color: #e67e22;
            """,
            "success": """
                background-color: #27ae60;
                color: #ffffff;
                border-color: #2ecc71;
            """,
            "error": """
                background-color: #c0392b;
                color: #ffffff;
                border-color: #e74c3c;
            """,
            "validating": """
                background-color: #8e44ad;
                color: #ffffff;
                border-color: #9b59b6;
            """
        }
        
        specific_style = styles.get(style_type, styles["info"])
        self.status_label.setStyleSheet(base_style + specific_style)
        self.status_label.setText(text)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header: Level Title
        self.title_label = QLabel("")
        self.title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Narrative Text
        self.story_text = QLabel("")
        self.story_text.setFont(QFont("Arial", 14))
        self.story_text.setWordWrap(True)
        self.story_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.story_text.setStyleSheet("background-color: #f8f9fa; padding: 15px; border-radius: 10px;")
        layout.addWidget(self.story_text)
        
        # Camera Feed
        self.camera_widget = CameraWidget()
        layout.addWidget(self.camera_widget)
        
        # Status/Instruction Bar
        self.status_label = QLabel("Say the spell when you're ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.set_status_message("Say the spell when you're ready", "info")
        layout.addWidget(self.status_label)
        
        # Progress Bar (for recording duration)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { background: #e0e0e0; border: none; } QProgressBar::chunk { background: #28a745; }")
        layout.addWidget(self.progress_bar)
        
        # Button Style
        button_style = """
            QPushButton {
                background-color: #5e1a1a;
                color: #f8f9fa;
                border-radius: 12px;
                border: 2px solid #f8f9fa;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #782020;
            }
            QPushButton:pressed {
                background-color: #441313;
            }
            QPushButton:disabled {
                background-color: #3d1111;
                color: #aaaaaa;
                border: 2px solid #aaaaaa;
            }
        """

        # Hidden buttons for touch support if needed (primary interaction is voice)
        self.controls_layout = QHBoxLayout()
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exit_btn.setStyleSheet(button_style)
        self.exit_btn.clicked.connect(self.exit_game)
        self.controls_layout.addWidget(self.exit_btn)
        
        self.next_btn = QPushButton("Next Level →")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self.next_level)
        self.next_btn.setVisible(False)
        self.next_btn.setStyleSheet(button_style)
        self.controls_layout.addWidget(self.next_btn)
        
        layout.addLayout(self.controls_layout)
        
        self.setLayout(layout)
        
        # Keep focus for accessibility/keyboard shortcuts
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def start_game(self):
        """Initialize game state."""
        self.story_manager.reset()
        self.level_completed = False
        self.is_listening = False
        self.game_active = False
        self.auto_retry_timer.stop()
        self.success_timer.stop()
        if self.verification_thread and self.verification_thread.isRunning():
            self.verification_thread.wait()
        if self.verification_thread:
            self.verification_thread.deleteLater()
            self.verification_thread = None
        
        # Init voice verifier
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                QMessageBox.warning(self, "API Key Missing", "Please set GOOGLE_API_KEY environment variable.")
                return
            self.voice_verifier = VoiceVerifier(api_key=api_key)
            self.camera_widget.start_camera()
            self.game_active = True
            self.load_step()
            self.setFocus()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start game: {str(e)}")

    def load_step(self):
        """Load current story step data."""
        step = self.story_manager.get_current_step()
        if not step:
            return

        self.title_label.setText(f"Chapter {step.id}: {step.title}")
        self.story_text.setText(step.description)
        self.level_completed = False
        self.is_listening = False
        self.auto_retry_timer.stop()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.success_timer.stop()

        # Reset spell engine visuals between steps
        if self.camera_widget.spell_engine:
            self.camera_widget.spell_engine.deactivate_spell()

        if step.step_type == StoryStepType.EXPLANATION:
            spell_name = step.required_spell.value if step.required_spell else "the spell"
            self.set_status_message(
                f"Spell briefing: study '{spell_name}' then tap Start.",
                "info"
            )
            self.next_btn.setText("Start →")
            self.next_btn.setVisible(True)
            self.next_btn.setEnabled(True)
        else:
            spell_name = step.required_spell.value if step.required_spell else "the spell"
            self.set_status_message(f"Say '{spell_name}' when you're ready.", "info")
            self.next_btn.setVisible(False)
            # Start automatic listening shortly after practice step loads
            self.schedule_spell_detection(delay_ms=1500)
    
    def schedule_spell_detection(self, delay_ms: int = 0):
        """Start or schedule automatic spell detection."""
        if (
            not self.voice_verifier
            or self.level_completed
            or not self.game_active
        ):
            return
        
        self.auto_retry_timer.stop()
        if delay_ms <= 0:
            self.start_listening()
        else:
            self.auto_retry_timer.start(delay_ms)
            
    def start_listening(self):
        if (
            not self.voice_verifier
            or self.is_listening
            or self.level_completed
            or not self.game_active
        ):
            return
        
        step = self.story_manager.get_current_step()
        if not step or step.step_type != StoryStepType.PRACTICE:
            return
        
        self.auto_retry_timer.stop()
        self.is_listening = True
        spell_name = step.required_spell.value
        self.set_status_message(f"Listening for '{spell_name}'... Speak clearly.", "listening")
        self.progress_bar.setRange(0, 0)

        self.verification_thread = SpellVerificationThread(self.voice_verifier, spell_name)
        self.verification_thread.recording_finished.connect(self.on_recording_finished)
        self.verification_thread.verification_complete.connect(self.on_verification_complete)
        self.verification_thread.start()

    def on_recording_finished(self):
        """Called when audio recording is done, but verification is still in progress."""
        self.set_status_message("Verifying spell...", "validating")

    def on_verification_complete(self, is_correct: bool, feedback: str):
        self.is_listening = False
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        
        if self.verification_thread:
            self.verification_thread.deleteLater()
            self.verification_thread = None
        
        if not self.game_active:
            return
        
        if is_correct:
            self.handle_practice_success(feedback)
        else:
            retry_feedback = feedback or "That wasn't quite right."
            self.set_status_message(
                f"{retry_feedback}\nI'm still listening—just say the spell again.",
                "error"
            )
            self.schedule_spell_detection(delay_ms=1200)

    def handle_practice_success(self, feedback: str):
        self.level_completed = True
        self.auto_retry_timer.stop()
        self.is_listening = False
        step = self.story_manager.get_current_step()
        if not step:
            return
        
        # Activate Visuals
        self.camera_widget.get_spell_engine().activate_spell(step.required_spell, None)
        
        # Update UI
        success_text = f"Success! {step.success_message}"
        if feedback:
            success_text = f"{success_text}\n{feedback}"
        self.set_status_message(success_text, "success")
        
        # Show Next button or Auto-advance
        if step.next_step_id:
            self.next_btn.setText("Next Spell →")
            self.next_btn.setVisible(True)
            self.next_btn.setEnabled(True)
            self.next_btn.setFocus()  # Move focus to next button
            # Auto-advance after brief celebration
            self.success_timer.start(3000)
        else:
            self.set_status_message("Congratulations! You have completed the game!", "success")
            QTimer.singleShot(5000, self.exit_game)

    def next_level(self):
        self.auto_retry_timer.stop()
        self.success_timer.stop()
        
        # Prevent skipping practice steps if not completed
        step = self.story_manager.get_current_step()
        if step and step.step_type == StoryStepType.PRACTICE and not self.level_completed:
            return

        self.is_listening = False
        if self.story_manager.advance_step():
            self.load_step()
        
    def exit_game(self):
        self.game_active = False
        self.auto_retry_timer.stop()
        self.success_timer.stop()
        if self.verification_thread and self.verification_thread.isRunning():
            self.verification_thread.wait()
        if self.verification_thread:
            self.verification_thread.deleteLater()
            self.verification_thread = None
        self.is_listening = False
        self.camera_widget.stop_camera()
        if self.voice_verifier:
            self.voice_verifier.release()
            self.voice_verifier = None
        self.finished.emit()

    def closeEvent(self, event):
        self.exit_game()
        super().closeEvent(event)

