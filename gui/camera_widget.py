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
from core.spell_engine import SpellEngine, SpellType
from core.object_identifier import ObjectIdentifier


class IdentificationThread(QThread):
    """Background thread for pattern identification."""
    
    identification_complete = pyqtSignal(str)  # Emits object name when done
    
    def __init__(self, object_identifier: ObjectIdentifier, pattern_image: np.ndarray):
        super().__init__()
        self.object_identifier = object_identifier
        self.pattern_image = pattern_image.copy()  # Make a copy to avoid issues
    
    def run(self):
        """Run identification in background."""
        try:
            object_name = self.object_identifier.identify_from_canvas(self.pattern_image)
            self.identification_complete.emit(object_name)
        except Exception as e:
            print(f"Error in identification thread: {e}")
            self.identification_complete.emit("wand")  # Default fallback


class CameraThread(QThread):
    """Thread for capturing and processing video frames."""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, hand_tracker: HandTracker, spell_engine: SpellEngine, object_identifier: Optional[ObjectIdentifier] = None):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.spell_engine = spell_engine
        self.object_identifier = object_identifier
        self.running = False
        self.cap = None
        # Smoothing state
        self.prev_wand_tip = None
        self.prev_wand_base = None
        self.smoothing_factor = 0.6  # Higher = more responsive, Lower = smoother
        # Identification state
        self.identification_thread: Optional[IdentificationThread] = None
        self.identification_requested = False
        
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
            
            # Reset identification_requested if spell changed or identification_pending is False
            if not hasattr(self, '_last_spell'):
                self._last_spell = None
            if (self._last_spell != self.spell_engine.current_spell or 
                (self.spell_engine.current_spell == SpellType.CURSOR and not self.spell_engine.identification_pending)):
                self.identification_requested = False
            self._last_spell = self.spell_engine.current_spell
            
            # Draw wand and get visual tip
            visual_tip = None
            if wand_tip:
                frame, visual_tip = self.spell_engine.draw_wand(frame, wand_tip, wand_base)
            
            # Draw spell effects
            # Use visual tip (extended wand) if available, otherwise use tracked tip
            effect_pos = visual_tip if visual_tip else (wand_tip_normalized if landmarks else None)
            
            # Debug: print effect_pos periodically
            if not hasattr(self, '_cam_debug_count'):
                self._cam_debug_count = 0
            self._cam_debug_count += 1
            if self._cam_debug_count % 30 == 0:
                print(f"[DEBUG CAM] visual_tip={visual_tip}, wand_tip_normalized={wand_tip_normalized}, effect_pos={effect_pos}")
            
            frame = self.spell_engine.draw_effects(frame, effect_pos)
            
            # Check if CURSOR spell needs identification
            if (self.spell_engine.current_spell == SpellType.CURSOR and 
                self.spell_engine.identification_pending and 
                not self.identification_requested and
                self.object_identifier is not None):
                
                self.identification_requested = True
                # Extract pattern image from cursor_path
                if len(self.spell_engine.cursor_path) > 1:
                    # Create a canvas image with the drawn path
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    canvas.fill(255)  # White background
                    
                    # Draw the path in black
                    pts = np.array(self.spell_engine.cursor_path, np.int32)
                    cv2.polylines(canvas, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
                    
                    # Start identification in background thread
                    self.identification_thread = IdentificationThread(self.object_identifier, canvas)
                    self.identification_thread.identification_complete.connect(self.on_identification_complete)
                    self.identification_thread.start()
            
            # Emit processed frame
            self.frame_ready.emit(frame)
            
            # Small delay to prevent overwhelming the system
            self.msleep(33)  # ~30 FPS
    
    def on_identification_complete(self, object_name: str):
        """Handle identification completion."""
        print(f"[DEBUG] Identification complete: {object_name}")
        self.spell_engine.identification_pending = False
        self.spell_engine.identified_object = object_name
        
        # Load the 3D model
        if self.spell_engine.load_cursor_model(object_name):
            print(f"[DEBUG] Model loaded successfully: {object_name}")
        else:
            print(f"[DEBUG] Failed to load model: {object_name}")
        
        # Clean up thread
        if self.identification_thread:
            self.identification_thread.wait()
            self.identification_thread = None
    
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
        
        # Initialize ObjectIdentifier (lazy initialization on first use)
        self.object_identifier: Optional[ObjectIdentifier] = None
        
    def start_camera(self, camera_index: int = 0):
        """Start camera capture."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        
        # Initialize ObjectIdentifier if not already done
        if self.object_identifier is None:
            try:
                import os
                api_key = os.getenv('GOOGLE_API_KEY')
                if api_key:
                    self.object_identifier = ObjectIdentifier(api_key=api_key)
                    print("[DEBUG] ObjectIdentifier initialized")
                else:
                    print("[WARNING] GOOGLE_API_KEY not set, pattern identification will not work")
            except Exception as e:
                print(f"[WARNING] Failed to initialize ObjectIdentifier: {e}")
        
        self.camera_thread = CameraThread(self.hand_tracker, self.spell_engine, self.object_identifier)
        self.camera_thread.frame_ready.connect(self.update_frame)
        
        try:
            self.camera_thread.start_capture(camera_index)
        except Exception as e:
            self.setText(f"Camera error: {str(e)}")
            raise
    
    def stop_camera(self):
        """Stop camera capture."""
        if self.camera_thread:
            # Wait for any identification thread to finish
            if self.camera_thread.identification_thread:
                self.camera_thread.identification_thread.wait()
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

