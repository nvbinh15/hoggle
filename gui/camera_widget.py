"""
Camera widget for displaying live video feed with AR overlays.
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from typing import Optional, Tuple
from core.hand_tracking import HandTracker
from core.spell_engine import SpellEngine


class CameraThread(QThread):
    """Thread for capturing and processing video frames."""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, hand_tracker: HandTracker, spell_engine: SpellEngine):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.spell_engine = spell_engine
        self.running = False
        self.cap = None
        
    def start_capture(self, camera_index: int = 0):
        """Start video capture."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        self.running = True
        self.start()
    
    def run(self):
        """Main thread loop."""
        import time
        last_time = time.time()
        
        while self.running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Process hand tracking
            landmarks = self.hand_tracker.process_frame(frame)
            
            # Get wand position
            wand_tip = None
            wand_base = None
            wand_tip_normalized = None
            if landmarks:
                # MediaPipe returns normalized coordinates (0-1)
                wand_tip_normalized = self.hand_tracker.get_wand_position(landmarks)
                wand_base_normalized = self.hand_tracker.get_wand_base(landmarks)
                
                # Convert to pixel coordinates for drawing
                h, w = frame.shape[:2]
                if wand_tip_normalized:
                    wand_tip = (
                        int(wand_tip_normalized[0] * w),
                        int(wand_tip_normalized[1] * h)
                    )
                if wand_base_normalized:
                    wand_base = (
                        int(wand_base_normalized[0] * w),
                        int(wand_base_normalized[1] * h)
                    )
            
            # Update spell engine (pass normalized coordinates)
            self.spell_engine.update(dt, wand_tip_normalized)
            
            # Draw wand
            if wand_tip:
                self.spell_engine.draw_wand(frame, wand_tip, wand_base)
            
            # Draw spell effects
            frame = self.spell_engine.draw_effects(frame, wand_tip_normalized if landmarks else None)
            
            # Emit processed frame
            self.frame_ready.emit(frame)
            
            # Small delay to prevent overwhelming the system
            self.msleep(33)  # ~30 FPS
    
    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class CameraWidget(QLabel):
    """Widget for displaying camera feed with AR overlays."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Camera not started")
        
        self.hand_tracker = HandTracker()
        # Load wand model from assets
        import os
        wand_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'wand.png')
        wand_path = os.path.abspath(wand_path)
        self.spell_engine = SpellEngine(640, 480, wand_model_path=wand_path)
        self.camera_thread = None
        
    def start_camera(self, camera_index: int = 0):
        """Start camera capture."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        
        self.camera_thread = CameraThread(self.hand_tracker, self.spell_engine)
        self.camera_thread.frame_ready.connect(self.update_frame)
        
        try:
            self.camera_thread.start_capture(camera_index)
        except Exception as e:
            self.setText(f"Camera error: {str(e)}")
            raise
    
    def stop_camera(self):
        """Stop camera capture."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
    
    def update_frame(self, frame: np.ndarray):
        """Update displayed frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
    
    def get_spell_engine(self) -> SpellEngine:
        """Get the spell engine instance."""
        return self.spell_engine
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.stop_camera()
        self.hand_tracker.release()
        super().closeEvent(event)

