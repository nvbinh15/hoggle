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
from PIL import Image, ImageDraw


class SpellType(Enum):
    """Types of spells available."""
    LUMOS = "Lumos"
    ACCIO = "Accio"
    WINGARDIUM_LEVIOSA = "Wingardium Leviosa"


class SpellEngine:
    """Manages spell visual effects and animations."""
    
    def __init__(self, frame_width: int, frame_height: int, wand_model_path: Optional[str] = None):
        """
        Initialize spell engine.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            wand_model_path: Path to wand GLB model file
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_spell: Optional[SpellType] = None
        self.spell_active = False
        self.animation_time = 0.0
        
        # Animation state
        self.accio_object_pos = None
        self.accio_target_pos = None
        self.leviosa_object_pos = None
        self.leviosa_hover_offset = 0.0
        
        # Load wand 2D sprite
        self.wand_sprite = None
        # Default size to scale to (optional, depends on PNG resolution)
        self.wand_sprite_size = (50, 150)  # Width, Height - 2x smaller than previous (100, 300)
        
        if wand_model_path is None:
            # Try default path
            default_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images', 'wand.png')
            wand_model_path = default_path if os.path.exists(default_path) else None
        
        if wand_model_path and os.path.exists(wand_model_path):
            print(f"Loading wand sprite from: {wand_model_path}")
            self._load_wand_sprite(wand_model_path)
    
    def _load_wand_sprite(self, sprite_path: str):
        """
        Load wand sprite from PNG file.
        
        Args:
            sprite_path: Path to PNG image file
        """
        try:
            # Load image with alpha channel
            img = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError("Could not load image")
            
            # Ensure 4 channels (BGRA)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
            # Flip horizontally to point inward
            img = cv2.flip(img, 1)
            
            # The wand image is diagonal (pointing top-left)
            # We need to rotate it to be vertical (pointing up)
            # Rotating 45 degrees clockwise
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            # Angle is negative for clockwise in some systems, but positive in cv2.getRotationMatrix2D is CCW
            # So for clockwise we need negative angle
            angle = -45 
            
            # Calculate new bounding box to avoid clipping
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            img = cv2.warpAffine(
                img, 
                rotation_matrix, 
                (new_w, new_h), 
                flags=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            self.wand_sprite = img
            print(f"Successfully loaded wand sprite of size {self.wand_sprite.shape}")
            
        except Exception as e:
            print(f"Error loading wand sprite from {sprite_path}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create a simple wand sprite
            self._create_fallback_wand_sprite()
    
    def _create_fallback_wand_sprite(self):
        """Create a simple fallback wand sprite if 3D model loading fails."""
        sprite_width, sprite_height = self.wand_sprite_size
        sprite_img = Image.new('RGBA', (sprite_width, sprite_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite_img)
        
        # Draw a simple brown wand
        center_x = sprite_width // 2
        wand_width = max(3, sprite_width // 20)
        
        # Wand body
        draw.ellipse(
            [(center_x - wand_width//2, 10), 
             (center_x + wand_width//2, sprite_height - 10)],
            fill=(139, 69, 19, 255)  # Brown
        )
        
        # Tip highlight
        draw.ellipse(
            [(center_x - wand_width//3, 10), 
             (center_x + wand_width//3, 20)],
            fill=(200, 200, 200, 255)  # Light gray
        )
        
        sprite_array = np.array(sprite_img)
        self.wand_sprite = cv2.cvtColor(sprite_array, cv2.COLOR_RGBA2BGRA)
        
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
        
        if spell_type == SpellType.ACCIO:
            # Start object at random edge
            edge = np.random.randint(0, 4)  # 0=top, 1=right, 2=bottom, 3=left
            if edge == 0:  # top
                self.accio_object_pos = (np.random.randint(0, self.frame_width), 0)
            elif edge == 1:  # right
                self.accio_object_pos = (self.frame_width, np.random.randint(0, self.frame_height))
            elif edge == 2:  # bottom
                self.accio_object_pos = (np.random.randint(0, self.frame_width), self.frame_height)
            else:  # left
                self.accio_object_pos = (0, np.random.randint(0, self.frame_height))
            
            self.accio_target_pos = wand_pixel
            
        elif spell_type == SpellType.WINGARDIUM_LEVIOSA:
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
        
        if self.current_spell == SpellType.ACCIO and self.accio_object_pos:
            # Update target position if wand moved
            if wand_pixel:
                self.accio_target_pos = wand_pixel
            
            if self.accio_target_pos:
                # Move object towards wand
                speed = 5.0  # pixels per frame (approximate)
                dx = self.accio_target_pos[0] - self.accio_object_pos[0]
                dy = self.accio_target_pos[1] - self.accio_object_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 10:
                    move_x = (dx / distance) * speed
                    move_y = (dy / distance) * speed
                    self.accio_object_pos = (
                        int(self.accio_object_pos[0] + move_x),
                        int(self.accio_object_pos[1] + move_y)
                    )
                else:
                    # Object reached wand
                    self.accio_object_pos = self.accio_target_pos
        
        elif self.current_spell == SpellType.WINGARDIUM_LEVIOSA and self.leviosa_object_pos:
            # Hover animation
            self.leviosa_hover_offset = math.sin(self.animation_time * 2.0) * 10
            if wand_pos:
                # Keep object near wand but hovering
                self.leviosa_object_pos = (
                    wand_pos[0] + 30,
                    int(wand_pos[1] - 50 + self.leviosa_hover_offset)
                )
    
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
        
        elif self.current_spell == SpellType.ACCIO:
            # Draw object moving towards wand
            if self.accio_object_pos:
                obj_size = 30
                cv2.rectangle(
                    frame,
                    (self.accio_object_pos[0] - obj_size//2, self.accio_object_pos[1] - obj_size//2),
                    (self.accio_object_pos[0] + obj_size//2, self.accio_object_pos[1] + obj_size//2),
                    (100, 150, 255),
                    -1
                )
                cv2.rectangle(
                    frame,
                    (self.accio_object_pos[0] - obj_size//2, self.accio_object_pos[1] - obj_size//2),
                    (self.accio_object_pos[0] + obj_size//2, self.accio_object_pos[1] + obj_size//2),
                    (255, 255, 255),
                    2
                )
        
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
                  wand_base: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw virtual wand on frame using 3D model sprite.
        
        Args:
            frame: BGR image frame
            wand_tip: Wand tip position (x, y) in pixel coordinates
            wand_base: Wand base position (x, y) in pixel coordinates
            
        Returns:
            Frame with wand drawn
        """
        if wand_tip is None:
            return frame
        
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
            base_pixel = (tip_pixel[0] - 20, tip_pixel[1] + 40)
        
        # Use 3D model sprite if available
        if self.wand_sprite is not None:
            # Calculate wand angle (from base to tip)
            dx = tip_pixel[0] - base_pixel[0]
            dy = tip_pixel[1] - base_pixel[1]
            angle = math.degrees(math.atan2(dy, dx)) + 90  # +90 because sprite is vertical (pointing up)
            
            sprite_h, sprite_w = self.wand_sprite.shape[:2]
            
            # We want to anchor the wand handle (bottom of sprite) to the base_pixel (hand)
            # So rotation center is bottom-center of sprite
            rotation_center = (sprite_w // 2, sprite_h)
            
            # Get rotation matrix around the handle
            rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
            
            # Calculate bounding box after rotation
            corners = np.array([
                [0, 0], [sprite_w, 0],
                [sprite_w, sprite_h], [0, sprite_h]
            ], dtype=np.float32)
            
            # Transform corners
            ones = np.ones((4, 1))
            corners_homogeneous = np.hstack([corners, ones])
            corners_rotated = (rotation_matrix @ corners_homogeneous.T).T
            
            # Get bounding box
            x_coords = corners_rotated[:, 0]
            y_coords = corners_rotated[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            # Canvas size
            canvas_w = x_max - x_min
            canvas_h = y_max - y_min
            
            # Adjust rotation matrix for canvas offset
            rotation_matrix[0, 2] += -x_min
            rotation_matrix[1, 2] += -y_min
            
            # Rotate sprite
            rotated_sprite = cv2.warpAffine(
                self.wand_sprite,
                rotation_matrix,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # Find where the handle ended up (should be at rotation_center after transform)
            handle_original = np.array([sprite_w // 2, sprite_h], dtype=np.float32)
            handle_after = np.array([sprite_w // 2 - x_min, sprite_h - y_min], dtype=int) 
            # Wait, the rotation center relative to the bounding box needs to be calculated correctly
            # Transform the handle point using the adjusted rotation matrix
            handle_homogeneous = np.array([sprite_w // 2, sprite_h, 1])
            handle_transformed = rotation_matrix @ handle_homogeneous
            handle_after = handle_transformed.astype(int)
            
            # Position sprite so handle aligns with base_pixel
            sprite_x = base_pixel[0] - handle_after[0]
            sprite_y = base_pixel[1] - handle_after[1]
            
            # Calculate overlay region
            x1 = max(0, sprite_x)
            y1 = max(0, sprite_y)
            x2 = min(frame.shape[1], sprite_x + canvas_w)
            y2 = min(frame.shape[0], sprite_y + canvas_h)
            
            if x2 > x1 and y2 > y1:
                # Calculate sprite region
                sprite_x1 = max(0, -sprite_x)
                sprite_y1 = max(0, -sprite_y)
                sprite_x2 = sprite_x1 + (x2 - x1)
                sprite_y2 = sprite_y1 + (y2 - y1)
                
                # Extract regions
                sprite_region = rotated_sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
                frame_region = frame[y1:y2, x1:x2]
                
                # Blend sprite onto frame using alpha channel
                if sprite_region.shape[:2] == frame_region.shape[:2] and sprite_region.shape[2] == 4:
                    alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
                    sprite_rgb = sprite_region[:, :, :3].astype(np.float32)
                    frame_region_float = frame_region.astype(np.float32)
                    frame_region[:, :] = ((1 - alpha) * frame_region_float + alpha * sprite_rgb).astype(np.uint8)
        
        else:
            # Fallback: draw wand as a line if sprite not available
            cv2.line(frame, base_pixel, tip_pixel, (139, 69, 19), 4)  # Brown color
            cv2.line(frame, base_pixel, tip_pixel, (101, 50, 14), 6)  # Darker brown outline
            cv2.circle(frame, tip_pixel, 3, (200, 200, 200), -1)
        
        return frame

