"""
Spell engine for managing visual effects of spells.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
import math
import time


class SpellType(Enum):
    """Types of spells available."""
    LUMOS = "Lumos"
    WINGARDIUM_LEVIOSA = "Wingardium Leviosa"


class SpellEngine:
    """Manages spell visual effects and animations."""
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize spell engine.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_spell: Optional[SpellType] = None
        self.spell_active = False
        self.animation_time = 0.0
        
        # Animation state
        self.leviosa_object_pos = None
        self.leviosa_hover_offset = 0.0
        
    def activate_spell(self, spell_type: SpellType, wand_pos: Optional[Tuple[float, float]] = None):
        """
        Activate a spell effect.
        
        Args:
            spell_type: Type of spell to activate
            wand_pos: Current wand position in normalized coordinates (0-1), or None for center
        """
        self.current_spell = spell_type
        self.spell_active = True
        self.animation_time = 0.0
        
        # Default to center if no wand position provided
        if wand_pos is None:
            wand_pos = (0.5, 0.5)
        
        # Convert normalized to pixel coordinates if needed
        wand_pixel = (
            int(wand_pos[0] * self.frame_width) if wand_pos[0] <= 1.0 else int(wand_pos[0]),
            int(wand_pos[1] * self.frame_height) if wand_pos[1] <= 1.0 else int(wand_pos[1])
        )
        
        if spell_type == SpellType.WINGARDIUM_LEVIOSA:
            # Start object near wand
            self.leviosa_object_pos = (wand_pixel[0] + 50, wand_pixel[1] - 50)
            self.leviosa_hover_offset = 0.0
    
    def deactivate_spell(self):
        """Deactivate current spell."""
        self.spell_active = False
        self.current_spell = None
    
    def update(self, dt: float, wand_pos: Optional[Tuple[float, float]] = None):
        """
        Update spell animations.
        
        Args:
            dt: Delta time since last update (seconds)
            wand_pos: Current wand position in normalized coordinates (0-1) for tracking
        """
        if not self.spell_active or self.current_spell is None:
            return
        
        self.animation_time += dt
        
        # Convert wand position to pixel coordinates if provided
        wand_pixel = None
        if wand_pos:
            if wand_pos[0] <= 1.0 and wand_pos[1] <= 1.0:
                wand_pixel = (
                    int(wand_pos[0] * self.frame_width),
                    int(wand_pos[1] * self.frame_height)
                )
            else:
                wand_pixel = (int(wand_pos[0]), int(wand_pos[1]))
        
        if self.current_spell == SpellType.WINGARDIUM_LEVIOSA and self.leviosa_object_pos:
            # Hover animation
            self.leviosa_hover_offset = math.sin(self.animation_time * 2.0) * 10
            if wand_pos:
                # Keep object near wand but hovering
                self.leviosa_object_pos = (
                    wand_pos[0] + 30,
                    int(wand_pos[1] - 50 + self.leviosa_hover_offset)
                )
    
    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions."""
        self.frame_width = width
        self.frame_height = height

    def draw_effects(self, frame: np.ndarray, wand_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw spell effects on frame.
        
        Args:
            frame: BGR image frame
            wand_pos: Current wand position (x, y) in normalized coordinates (0-1)
            
        Returns:
            Frame with effects drawn
        """
        if not self.spell_active or self.current_spell is None:
            return frame
        
        # Convert normalized wand_pos to pixel coordinates if needed
        if wand_pos:
            if wand_pos[0] <= 1.0 and wand_pos[1] <= 1.0:
                # Normalized coordinates
                wand_pixel = (
                    int(wand_pos[0] * self.frame_width),
                    int(wand_pos[1] * self.frame_height)
                )
            else:
                # Already pixel coordinates
                wand_pixel = wand_pos
        else:
            wand_pixel = None
        
        if self.current_spell == SpellType.LUMOS:
            # Draw glowing circle at wand tip
            if wand_pixel:
                glow_radius = int(15 + 5 * math.sin(self.animation_time * 5.0))
                cv2.circle(frame, wand_pixel, glow_radius, (255, 255, 200), -1)
                cv2.circle(frame, wand_pixel, glow_radius + 5, (255, 255, 100), 2)
        
        elif self.current_spell == SpellType.WINGARDIUM_LEVIOSA:
            # Draw floating object
            if self.leviosa_object_pos:
                obj_size = 25
                cv2.ellipse(
                    frame,
                    self.leviosa_object_pos,
                    (obj_size, obj_size//2),
                    0,
                    0,
                    360,
                    (200, 200, 100),
                    -1
                )
                cv2.ellipse(
                    frame,
                    self.leviosa_object_pos,
                    (obj_size, obj_size//2),
                    0,
                    0,
                    360,
                    (255, 255, 200),
                    2
                )
        
        return frame
    
    def draw_wand(self, frame: np.ndarray, wand_tip: Optional[Tuple[int, int]], 
                  wand_base: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Draw virtual wand on frame using programmatic drawing.
        
        Args:
            frame: BGR image frame
            wand_tip: Wand tip position (x, y) in pixel coordinates
            wand_base: Wand base position (x, y) in pixel coordinates
            
        Returns:
            Tuple of (Frame with wand drawn, visual tip position (x, y))
        """
        if wand_tip is None:
            return frame, None
        
        # Convert normalized coordinates if needed
        if wand_tip[0] <= 1.0 and wand_tip[1] <= 1.0:
            tip_pixel = (
                int(wand_tip[0] * self.frame_width),
                int(wand_tip[1] * self.frame_height)
            )
        else:
            tip_pixel = wand_tip
        
        if wand_base:
            if wand_base[0] <= 1.0 and wand_base[1] <= 1.0:
                base_pixel = (
                    int(wand_base[0] * self.frame_width),
                    int(wand_base[1] * self.frame_height)
                )
            else:
                base_pixel = wand_base
        else:
            # Default base position (slightly offset from tip)
            base_pixel = (tip_pixel[0] - 100, tip_pixel[1] + 200)
            
        # Calculate wand vector
        dx = tip_pixel[0] - base_pixel[0]
        dy = tip_pixel[1] - base_pixel[1]
        original_length = math.sqrt(dx*dx + dy*dy)
        
        if original_length < 1.0:
            return frame, None
            
        # Normalize vector
        ux = dx / original_length
        uy = dy / original_length
        
        # Extend wand length (make it visually longer)
        # The original length is just the finger length (base to tip)
        # We extend it to look like a real wand
        extended_length = original_length * 5.0
        
        # Recalculate tip pixel based on extended length
        tip_pixel = (
            int(base_pixel[0] + ux * extended_length),
            int(base_pixel[1] + uy * extended_length)
        )
        
        # Use extended length for drawing calculations
        length = extended_length
        
        # Perpendicular vector
        px = -uy
        py = ux
        
        # Wand parameters
        width_base = 30
        width_tip = 10
        
        # Calculate polygon corners
        # Base corners
        b1x = base_pixel[0] + px * width_base
        b1y = base_pixel[1] + py * width_base
        b2x = base_pixel[0] - px * width_base
        b2y = base_pixel[1] - py * width_base
        
        # Tip corners
        t1x = tip_pixel[0] + px * width_tip
        t1y = tip_pixel[1] + py * width_tip
        t2x = tip_pixel[0] - px * width_tip
        t2y = tip_pixel[1] - py * width_tip
        
        # Create polygon points
        pts = np.array([
            [b1x, b1y],
            [t1x, t1y],
            [t2x, t2y],
            [b2x, b2y]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw wand body (dark brown)
        cv2.fillPoly(frame, [pts], (30, 50, 90)) # BGR: Dark Brown
        
        # Draw wand border (lighter brown)
        cv2.polylines(frame, [pts], True, (50, 80, 120), 2, cv2.LINE_AA)
        
        # Draw handle detail (lighter grip at base)
        handle_len = length * 0.25
        hx = base_pixel[0] + ux * handle_len
        hy = base_pixel[1] + uy * handle_len
        
        # Interpolate width at handle end
        width_handle = width_base - ((width_base - width_tip) * 0.25)
        
        h1x = hx + px * width_handle
        h1y = hy + py * width_handle
        h2x = hx - px * width_handle
        h2y = hy - py * width_handle
        
        handle_pts = np.array([
            [b1x, b1y],
            [h1x, h1y],
            [h2x, h2y],
            [b2x, b2y]
        ], np.int32)
        handle_pts = handle_pts.reshape((-1, 1, 2))
        
        # Draw handle overlay
        cv2.fillPoly(frame, [handle_pts], (40, 70, 110)) 
        cv2.polylines(frame, [handle_pts], True, (60, 90, 140), 2, cv2.LINE_AA)
        
        # Draw tip highlight
        cv2.circle(frame, tip_pixel, 5, (200, 200, 200), -1)
        
        return frame, tip_pixel

