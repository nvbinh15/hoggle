"""
Hand tracking module using MediaPipe for wand detection.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class HandTracker:
    """Tracks hand landmarks and calculates wand position."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """
        Process a frame and return hand landmarks.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            List of landmark coordinates if hand detected, None otherwise
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Return the first hand's landmarks
            return results.multi_hand_landmarks[0]
        return None
    
    def get_wand_position(self, landmarks) -> Optional[Tuple[float, float]]:
        """
        Calculate wand tip position from hand landmarks.
        Wand is positioned at the tip of the index finger.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            (x, y) tuple of wand tip position in normalized coordinates (0-1), or None
        """
        if landmarks is None:
            return None
        
        # Get index finger tip (landmark 8)
        # MediaPipe returns normalized coordinates (0-1)
        index_tip = landmarks.landmark[8]
        return (index_tip.x, index_tip.y)
    
    def get_wand_base(self, landmarks) -> Optional[Tuple[float, float]]:
        """
        Calculate wand base position (where wand starts from hand).
        Uses the middle finger MCP joint as the base.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            (x, y) tuple of wand base position in normalized coordinates (0-1), or None
        """
        if landmarks is None:
            return None
        
        # Get middle finger MCP (landmark 9)
        # MediaPipe returns normalized coordinates (0-1)
        middle_mcp = landmarks.landmark[9]
        return (middle_mcp.x, middle_mcp.y)
    
    def draw_hand_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw hand landmarks on frame for debugging.
        
        Args:
            frame: BGR image frame
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        return frame
    
    def release(self):
        """Release resources."""
        self.hands.close()

