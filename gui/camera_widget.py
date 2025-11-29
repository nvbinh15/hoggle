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
        # Smoothing state
        self.prev_wand_tip = None
        self.prev_wand_base = None
        self.smoothing_factor = 0.6  # Higher = more responsive, Lower = smoother
        
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
            
            # Update frame size in spell engine
            h, w = frame.shape[:2]
            self.spell_engine.update_frame_size(w, h)
            
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
                raw_tip_norm = self.hand_tracker.get_wand_position(landmarks)
                raw_base_norm = self.hand_tracker.get_wand_base(landmarks)
                
                # Apply smoothing to Tip
                if raw_tip_norm:
                    curr_tip_x = raw_tip_norm[0] * w
                    curr_tip_y = raw_tip_norm[1] * h
                    
                    if self.prev_wand_tip:
                        smooth_x = curr_tip_x * self.smoothing_factor + self.prev_wand_tip[0] * (1 - self.smoothing_factor)
                        smooth_y = curr_tip_y * self.smoothing_factor + self.prev_wand_tip[1] * (1 - self.smoothing_factor)
                        self.prev_wand_tip = (smooth_x, smooth_y)
                    else:
                        self.prev_wand_tip = (curr_tip_x, curr_tip_y)
                    
                    wand_tip = (int(self.prev_wand_tip[0]), int(self.prev_wand_tip[1]))
                    wand_tip_normalized = (self.prev_wand_tip[0] / w, self.prev_wand_tip[1] / h)
                
                # Apply smoothing to Base
                if raw_base_norm:
                    curr_base_x = raw_base_norm[0] * w
                    curr_base_y = raw_base_norm[1] * h
                    
                    if self.prev_wand_base:
                        smooth_x = curr_base_x * self.smoothing_factor + self.prev_wand_base[0] * (1 - self.smoothing_factor)
                        smooth_y = curr_base_y * self.smoothing_factor + self.prev_wand_base[1] * (1 - self.smoothing_factor)
                        self.prev_wand_base = (smooth_x, smooth_y)
                    else:
                        self.prev_wand_base = (curr_base_x, curr_base_y)
                    
                    wand_base = (int(self.prev_wand_base[0]), int(self.prev_wand_base[1]))
            else:
                # Reset smoothing if hand lost
                self.prev_wand_tip = None
                self.prev_wand_base = None
            
            # Update spell engine (pass normalized coordinates)
            self.spell_engine.update(dt, wand_tip_normalized)
            
            # Draw wand and get visual tip
            visual_tip = None
            if wand_tip:
                frame, visual_tip = self.spell_engine.draw_wand(frame, wand_tip, wand_base)
            
            # Draw spell effects
            # Use visual tip (extended wand) if available, otherwise use tracked tip
            effect_pos = visual_tip if visual_tip else (wand_tip_normalized if landmarks else None)
            frame = self.spell_engine.draw_effects(frame, effect_pos)
            
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
        self.spell_engine = SpellEngine(640, 480)
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

