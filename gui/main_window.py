"""
Main window for Hoggle application.
"""
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QStyleOption, QStyle,
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QUrl
from PyQt6.QtGui import QFont, QPainter, QPixmap, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui.story_screen import StoryScreen


class BackgroundImageScreen(QWidget):
    """Utility widget that paints the Hogwarts background image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        image_path = os.path.join(os.getcwd(), "assets", "images", "hogwarts.jpeg")
        self.background_image = QPixmap(image_path)
        if self.background_image.isNull():
            print("Warning: Failed to load hogwarts.jpeg background.")

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        try:
            self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
            if not self.background_image.isNull():
                painter.drawPixmap(self.rect(), self.background_image)
        finally:
            painter.end()


class MainMenuScreen(BackgroundImageScreen):
    """Welcome screen with program branding and entry button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(40)
        layout.setContentsMargins(80, 60, 80, 60)

        layout.addStretch()

        title = QLabel("Hoggle - Hogwarts for Muggle")
        title.setFont(QFont("Arial", 40, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ffffff;")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(35)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 180))
        title.setGraphicsEffect(shadow)
        layout.addWidget(title)

        layout.addStretch()

        start_btn = QPushButton("Start")
        start_btn.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        start_btn.setMinimumHeight(80)
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e1a1a;
                color: #f8f9fa;
                border-radius: 12px;
                border: 2px solid #f8f9fa;
            }
            QPushButton:hover {
                background-color: #782020;
            }
            QPushButton:pressed {
                background-color: #441313;
            }
        """)
        start_btn.clicked.connect(self.start_intro_sequence)
        layout.addWidget(start_btn)

        layout.addStretch()
        self.setLayout(layout)

    def start_intro_sequence(self):
        if self.parent_window:
            self.parent_window.show_intro_sequence()


class IntroSequenceScreen(BackgroundImageScreen):
    """Full-screen message sequence before starting spells."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.messages = [
            "Welcome to Hoggle — Hogwarts orientation for muggles who still call it Levio-SAH.",
            "You'll be guided spell-by-spell. Each chapter introduces a spell, then lets you practice it.",
            "Focus, breathe, and follow the narrator. The next spell only appears when you've mastered the current one."
        ]
        self.current_index = 0
        self.active_animations = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(120, 80, 120, 80)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setFont(QFont("Arial", 16, QFont.Weight.DemiBold))
        self.progress_label.setStyleSheet("color: #f8f9fa;")
        layout.addWidget(self.progress_label)

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.message_label.setStyleSheet("""
            color: #f8f9fa;
            background-color: rgba(0, 0, 0, 0.45);
            padding: 35px;
            border-radius: 18px;
        """)
        layout.addWidget(self.message_label, stretch=1)

        self.progress_opacity = QGraphicsOpacityEffect(self)
        self.progress_label.setGraphicsEffect(self.progress_opacity)
        self.progress_opacity.setOpacity(0)

        self.message_opacity = QGraphicsOpacityEffect(self)
        self.message_label.setGraphicsEffect(self.message_opacity)
        self.message_opacity.setOpacity(0)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.next_button = QPushButton("Next")
        self.next_button.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.next_button.setMinimumWidth(200)
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #f8f9fa;
                color: #5e1a1a;
                border-radius: 10px;
                padding: 14px 24px;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
        """)
        self.next_button.clicked.connect(self.advance_message)
        button_row.addWidget(self.next_button)

        button_row.addStretch()
        layout.addLayout(button_row)

        self.setLayout(layout)

    def start_sequence(self):
        self.current_index = 0
        self.update_message(initial=True)

    def update_message(self, initial=False):
        total = len(self.messages)
        next_progress = f"{self.current_index + 1} / {total}"
        next_message = self.messages[self.current_index]

        if initial:
            self.progress_label.setText(next_progress)
            self.message_label.setText(next_message)
            self.progress_opacity.setOpacity(0)
            self.message_opacity.setOpacity(0)
            self._fade_in_effect(self.progress_opacity, duration=400)
            self._fade_in_effect(self.message_opacity, duration=600)
        else:
            self._animate_label_update(self.progress_label, self.progress_opacity, next_progress)
            self._animate_label_update(self.message_label, self.message_opacity, next_message)

        if self.current_index == total - 1:
            self.next_button.setText("Begin")
        else:
            self.next_button.setText("Next")

    def advance_message(self):
        if self.current_index < len(self.messages) - 1:
            self.current_index += 1
            self.update_message()
        else:
            if self.parent_window:
                self.parent_window.start_story_mode()

    def _fade_in_effect(self, effect: QGraphicsOpacityEffect, duration: int = 450):
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(duration)
        animation.setStartValue(effect.opacity())
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        self._track_animation(animation)

    def _animate_label_update(
        self,
        label: QLabel,
        effect: QGraphicsOpacityEffect,
        new_text: str,
        fade_out_duration: int = 220,
        fade_in_duration: int = 420
    ):
        if effect is None:
            label.setText(new_text)
            return

        fade_out = QPropertyAnimation(effect, b"opacity", self)
        fade_out.setDuration(fade_out_duration)
        fade_out.setStartValue(effect.opacity())
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        fade_in = QPropertyAnimation(effect, b"opacity", self)
        fade_in.setDuration(fade_in_duration)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        def start_fade_in():
            label.setText(new_text)
            fade_in.start()

        fade_out.finished.connect(start_fade_in)

        self._track_animation(fade_out)
        self._track_animation(fade_in)

        if effect.opacity() <= 0.05:
            start_fade_in()
        else:
            fade_out.start()

    def _track_animation(self, animation: QPropertyAnimation):
        if animation is None:
            return

        self.active_animations.append(animation)

        def cleanup():
            if animation in self.active_animations:
                self.active_animations.remove(animation)

        animation.finished.connect(cleanup)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("hoggle - hogwarts for muggle")
        self.setMinimumSize(1024, 768)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_menu_screen = MainMenuScreen(self)
        self.intro_sequence_screen = IntroSequenceScreen(self)
        self.story_screen = StoryScreen(self)

        self.story_screen.finished.connect(self.show_main_menu)

        self.stacked_widget.addWidget(self.main_menu_screen)
        self.stacked_widget.addWidget(self.intro_sequence_screen)
        self.stacked_widget.addWidget(self.story_screen)

        # Background Music Setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # Set volume (0.0 to 1.0)
        self.audio_output.setVolume(0.3)
        
        # Load music file
        music_path = os.path.join(os.getcwd(), "assets", "sound", "background-music.mp3")
        if os.path.exists(music_path):
            self.player.setSource(QUrl.fromLocalFile(music_path))
            self.player.setLoops(QMediaPlayer.Loops.Infinite)
            self.player.play()
        else:
            print(f"Warning: Music file not found at {music_path}")

        self.show_main_menu()

    def show_main_menu(self):
        """Show welcome screen."""
        # Resume music if it was paused
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.play()
        
        self.stacked_widget.setCurrentWidget(self.main_menu_screen)

    def show_intro_sequence(self):
        """Display intro messages before spells."""
        self.intro_sequence_screen.start_sequence()
        self.stacked_widget.setCurrentWidget(self.intro_sequence_screen)

    def start_story_mode(self):
        """Start the spell guidance story."""
        # Pause music when entering practice mode (camera active)
        self.player.pause()
        
        self.stacked_widget.setCurrentWidget(self.story_screen)
        self.story_screen.start_game()

    def closeEvent(self, event):
        """Clean up on close."""
        if self.story_screen:
            self.story_screen.exit_game()
        super().closeEvent(event)
