"""
Spell engine for managing visual effects of spells.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
import math
import time
import os


class SpellType(Enum):
    """Types of spells available."""
    LUMOS = "Lumos"
    WINGARDIUM_LEVIOSA = "Wingardium Leviosa"
    CURSOR = "Cursor"


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
        self.leviosa_state = 'idle'  # idle, grounded, floating
        self.leviosa_object_pos = None
        self.leviosa_hover_offset = 0.0
        
        # CURSOR spell state
        self.cursor_path: List[Tuple[int, int]] = []
        self.recording_start_time: Optional[float] = None
        self.identification_pending: bool = False
        self.identified_object: Optional[str] = None
        self.cursor_image: Optional[np.ndarray] = None
        self.cursor_model_rotation: float = 0.0
        self.cursor_model_position: Optional[Tuple[int, int]] = None
        
    def setup_spell_scene(self, spell_type: SpellType):
        """
        Setup the visual scene for a spell (before activation).
        
        Args:
            spell_type: Type of spell to setup
        """
        print(f"[DEBUG] setup_spell_scene called: spell_type={spell_type}")
        self.current_spell = spell_type
        self.animation_time = 0.0
        
        # Reset specific spell states
        if spell_type == SpellType.CURSOR:
            self.recording_start_time = None
            self.cursor_path = []
            self.identified_object = None
            self.cursor_image = None
        
        if spell_type == SpellType.WINGARDIUM_LEVIOSA:
            # Start object at bottom center
            self.leviosa_state = 'grounded'
            center_x = self.frame_width // 2
            bottom_y = self.frame_height - 50
            self.leviosa_object_pos = (center_x, bottom_y)
            self.leviosa_hover_offset = 0.0
            # For Leviosa, we need spell_active=True to show the grounded box
            self.spell_active = True
        else:
            # For other spells (Lumos, Cursor), wait for activation to show effects
            self.spell_active = False

    def activate_spell(self, spell_type: SpellType, wand_pos: Optional[Tuple[float, float]] = None):
        """
        Activate a spell effect.
        
        Args:
            spell_type: Type of spell to activate
            wand_pos: Current wand position in normalized coordinates (0-1), or None for center
        """
        print(f"[DEBUG] activate_spell called: spell_type={spell_type}, wand_pos={wand_pos}")
        self.current_spell = spell_type
        self.spell_active = True
        
        # If it's a new spell activation (not just scene setup), reset time
        if spell_type != SpellType.WINGARDIUM_LEVIOSA or self.leviosa_state == 'idle':
             self.animation_time = 0.0

        print(f"[DEBUG] Spell activated: spell_active={self.spell_active}, current_spell={self.current_spell}")
        
        # Default to center if no wand position provided
        if wand_pos is None:
            wand_pos = (0.5, 0.5)
        
        # Convert normalized to pixel coordinates if needed
        wand_pixel = (
            int(wand_pos[0] * self.frame_width) if wand_pos[0] <= 1.0 else int(wand_pos[0]),
            int(wand_pos[1] * self.frame_height) if wand_pos[1] <= 1.0 else int(wand_pos[1])
        )
        
        if spell_type == SpellType.WINGARDIUM_LEVIOSA:
            # Trigger float animation
            self.leviosa_state = 'floating'
        elif spell_type == SpellType.CURSOR:
            # Initialize CURSOR recording state
            self.cursor_path = []
            self.recording_start_time = time.time()
            self.identification_pending = False
            self.identified_object = None
            self.cursor_image = None
            self.cursor_model_rotation = 0.0
            self.cursor_model_position = None
    
    def deactivate_spell(self):
        """Deactivate current spell."""
        print(f"[DEBUG] deactivate_spell called")
        self.spell_active = False
        self.current_spell = None
        self.leviosa_state = 'idle'
    
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
            if self.leviosa_state == 'floating':
                # Move up until we reach target height (e.g., 1/3 of screen)
                target_y = self.frame_height // 3
                current_x, current_y = self.leviosa_object_pos
                
                # Rising speed
                rise_speed = 100 * dt # pixels per second
                
                if current_y > target_y:
                    current_y -= rise_speed
                
                # Hover animation
                self.leviosa_hover_offset = math.sin(self.animation_time * 2.0) * 10
                
                # If tracking wand, could pull towards wand x
                # For now, keep simple vertical rise + hover
                
                self.leviosa_object_pos = (
                    current_x,
                    int(current_y)
                )
            elif self.leviosa_state == 'grounded':
                # Ensure it stays at bottom (in case of resize)
                self.leviosa_object_pos = (
                    self.frame_width // 2,
                    self.frame_height - 50
                )
        elif self.current_spell == SpellType.CURSOR:
            # CURSOR spell logic
            if self.recording_start_time is not None:
                elapsed = time.time() - self.recording_start_time
                
                # Recording phase: first 5 seconds
                if elapsed < 5.0:
                    # Record wand tip positions
                    if wand_pixel:
                        self.cursor_path.append(wand_pixel)
                # After 5 seconds, trigger identification
                elif not self.identification_pending and self.identified_object is None:
                    self.identification_pending = True
                    # Set position for model display (center of path or last position)
                    if self.cursor_path:
                        # Use center of bounding box of path
                        xs = [p[0] for p in self.cursor_path]
                        ys = [p[1] for p in self.cursor_path]
                        if xs and ys:
                            self.cursor_model_position = (
                                int(sum(xs) / len(xs)),
                                int(sum(ys) / len(ys))
                            )
                    else:
                        self.cursor_model_position = wand_pixel if wand_pixel else (self.frame_width // 2, self.frame_height // 2)
            
            # Display phase: animate 3D model rotation
            if self.identified_object and self.cursor_image is not None:
                self.cursor_model_rotation += dt * 1.0  # Rotate 1 radian per second
    
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
        # Debug: print every ~30 frames to avoid spam
        if hasattr(self, '_debug_frame_count'):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 0
        
        should_debug = (self._debug_frame_count % 30 == 0)
        
        if should_debug:
            print(f"[DEBUG] draw_effects: spell_active={self.spell_active}, current_spell={self.current_spell}, wand_pos={wand_pos}")
        
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
                wand_pixel = (int(wand_pos[0]), int(wand_pos[1]))
        else:
            wand_pixel = None
        
        if should_debug:
            print(f"[DEBUG] draw_effects after conversion: wand_pixel={wand_pixel}")
        
        if self.current_spell == SpellType.LUMOS:
            # Draw glowing circle at wand tip
            if wand_pixel:
                # Increased base radius and amplitude for bigger, more dynamic glow
                glow_radius = int(30 + 10 * math.sin(self.animation_time * 5.0))
                if should_debug:
                    print(f"[DEBUG] Drawing LUMOS glow at {wand_pixel} with radius {glow_radius}")
                
                # Draw multiple layers for a more intense, brighter glow effect
                # Outer glow layer (largest, most transparent)
                cv2.circle(frame, wand_pixel, glow_radius + 20, (255, 255, 150), -1)
                # Middle glow layer
                cv2.circle(frame, wand_pixel, glow_radius + 10, (255, 255, 200), -1)
                # Inner bright core (pure white)
                cv2.circle(frame, wand_pixel, glow_radius, (255, 255, 255), -1)
                # Bright outer ring for extra visibility
                cv2.circle(frame, wand_pixel, glow_radius + 15, (255, 255, 180), 3)
            elif should_debug:
                print(f"[DEBUG] LUMOS spell active but wand_pixel is None - cannot draw glow")
        
        elif self.current_spell == SpellType.WINGARDIUM_LEVIOSA:
            # Draw box object
            if self.leviosa_object_pos:
                box_size = 60
                x, y = int(self.leviosa_object_pos[0]), int(self.leviosa_object_pos[1])
                
                # Add hover offset if floating
                if self.leviosa_state == 'floating':
                    y += int(self.leviosa_hover_offset)
                
                # Draw a crate-like box
                # Main box body
                top_left = (x - box_size//2, y - box_size//2)
                bottom_right = (x + box_size//2, y + box_size//2)
                
                # Fill - brown wood color
                cv2.rectangle(frame, top_left, bottom_right, (30, 70, 110), -1)
                
                # Border - lighter wood
                cv2.rectangle(frame, top_left, bottom_right, (50, 90, 130), 3)
                
                # Cross pattern
                cv2.line(frame, top_left, bottom_right, (40, 80, 120), 2)
                cv2.line(frame, (x + box_size//2, y - box_size//2), (x - box_size//2, y + box_size//2), (40, 80, 120), 2)
                
                # Inner border
                inset = 5
                cv2.rectangle(frame, (top_left[0]+inset, top_left[1]+inset), (bottom_right[0]-inset, bottom_right[1]-inset), (45, 85, 125), 1)

        elif self.current_spell == SpellType.CURSOR:
            # CURSOR spell drawing
            if self.recording_start_time is not None:
                elapsed = time.time() - self.recording_start_time
                
                # Recording phase: draw path
                if elapsed < 5.0:
                    if len(self.cursor_path) > 1:
                        # Draw polyline path
                        pts = np.array(self.cursor_path, np.int32)
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 3, cv2.LINE_AA)
                        # Draw current point
                        if self.cursor_path:
                            cv2.circle(frame, self.cursor_path[-1], 5, (255, 255, 0), -1)
                
                # Display phase: render image
                if self.identified_object and self.cursor_image is not None and self.cursor_model_position:
                    # Draw the image
                    x, y = self.cursor_model_position
                    h, w = self.cursor_image.shape[:2]
                    
                    # Calculate top-left position
                    top_left_x = int(x - w / 2)
                    top_left_y = int(y - h / 2)
                    
                    # Handle overlay with alpha channel
                    self._overlay_image(frame, self.cursor_image, top_left_x, top_left_y)
                    
                    # Draw object name
                    text = self.identified_object.capitalize()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Position text above the object
                    text_x = int(x - text_size[0] / 2)
                    text_y = int(y - h / 2 - 20) # Above object with padding
                    
                    # Draw text outline (black) for better visibility
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    # Draw text (white)
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame
    
    def load_cursor_model(self, object_name: str) -> bool:
        """
        Load a 2D image for the CURSOR spell.
        
        Args:
            object_name: Name of the object (ball, cat, heart, pizza, star, wand)
            
        Returns:
            True if image loaded successfully, False otherwise
        """
        # Try different extensions
        extensions = ['.png', '.jpg', '.jpeg']
        image_path = None
        
        for ext in extensions:
            path = os.path.join(os.getcwd(), "assets", "images", f"{object_name}{ext}")
            if os.path.exists(path):
                image_path = path
                break
                
        if not image_path:
            print(f"Image file not found for object: {object_name}")
            return False
        
        try:
            # Load image with alpha channel if possible
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                return False
                
            # Resize to reasonable size (max dimension ~300px)
            h, w = image.shape[:2]
            max_dim = max(h, w)
            if max_dim > 300:
                scale = 300.0 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            self.cursor_image = image
            print(f"Loaded image: {object_name}")
            return True
        except Exception as e:
            print(f"Error loading image {object_name}: {e}")
            return False
    
    def _overlay_image(self, background: np.ndarray, foreground: np.ndarray, x: int, y: int) -> None:
        """
        Overlay a foreground image onto a background image at (x, y) handling alpha channel.
        
        Args:
            background: Background image (BGR) - modified in place
            foreground: Foreground image (BGRA or BGR)
            x: Top-left x coordinate
            y: Top-left y coordinate
        """
        h_fg, w_fg = foreground.shape[:2]
        h_bg, w_bg = background.shape[:2]
        
        if x >= w_bg or y >= h_bg:
            return
            
        # Crop foreground if it goes outside background
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(w_bg, x + w_fg)
        y_end = min(h_bg, y + h_fg)
        
        # Calculate source coordinates
        fg_x_start = x_start - x
        fg_y_start = y_start - y
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)
        
        if fg_x_end <= fg_x_start or fg_y_end <= fg_y_start:
            return
            
        fg_crop = foreground[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
        bg_crop = background[y_start:y_end, x_start:x_end]
        
        # Check if foreground has alpha channel
        if fg_crop.shape[2] == 4:
            alpha = fg_crop[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            for c in range(3):
                bg_crop[:, :, c] = (alpha * fg_crop[:, :, c] + alpha_inv * bg_crop[:, :, c])
        else:
            background[y_start:y_end, x_start:x_end] = fg_crop
    
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

