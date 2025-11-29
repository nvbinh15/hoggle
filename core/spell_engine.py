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
import trimesh


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
        self.model_mesh: Optional[trimesh.Trimesh] = None
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
            self.model_mesh = None
        
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
            self.model_mesh = None
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
            if self.identified_object and self.model_mesh:
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
                
                # Display phase: render 3D model
                if self.identified_object and self.model_mesh and self.cursor_model_position:
                    self._render_3d_model(frame, self.model_mesh, self.cursor_model_position, self.cursor_model_rotation)
                    
                    # Draw object name
                    text = self.identified_object.capitalize()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Position text above the object
                    text_x = int(self.cursor_model_position[0] - text_size[0] / 2)
                    text_y = int(self.cursor_model_position[1] - 180) # Above object (radius ~150 + padding)
                    
                    # Draw text outline (black) for better visibility
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    # Draw text (white)
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame
    
    def load_cursor_model(self, object_name: str) -> bool:
        """
        Load a 3D model for the CURSOR spell.
        
        Args:
            object_name: Name of the object (ball, cat, heart, pizza, star, wand)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = os.path.join(os.getcwd(), "assets", "3d", f"{object_name}.glb")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            # Load GLB file using trimesh
            scene = trimesh.load(model_path)
            
            # Extract mesh from scene (GLB files are scenes)
            if isinstance(scene, trimesh.Scene):
                # Get the first geometry from the scene
                if len(scene.geometry) > 0:
                    self.model_mesh = list(scene.geometry.values())[0]
                else:
                    print("No geometry found in scene")
                    return False
            elif isinstance(scene, trimesh.Trimesh):
                self.model_mesh = scene
            else:
                print(f"Unexpected scene type: {type(scene)}")
                return False
            
            # Center and normalize the mesh
            if self.model_mesh is not None:
                # Center the mesh
                self.model_mesh.vertices -= self.model_mesh.vertices.mean(axis=0)
                # Scale to reasonable size (normalize to fit in ~200 pixel radius)
                max_dim = self.model_mesh.vertices.max(axis=0) - self.model_mesh.vertices.min(axis=0)
                max_extent = max(max_dim)
                if max_extent > 0:
                    scale = 300.0 / max_extent  # Scale to ~300 pixel radius (3x bigger)
                    self.model_mesh.vertices *= scale
            
            print(f"Loaded model: {object_name}")
            return True
        except Exception as e:
            print(f"Error loading model {object_name}: {e}")
            return False
    
    def _render_3d_model(self, frame: np.ndarray, mesh: trimesh.Trimesh, 
                        position: Tuple[int, int], rotation: float) -> None:
        """
        Software renderer for 3D models using Painter's algorithm.
        
        Args:
            frame: BGR image frame to draw on
            mesh: Trimesh object to render
            position: Center position (x, y) in pixel coordinates
            rotation: Rotation angle in radians around Z axis
        """
        if mesh is None or len(mesh.vertices) == 0:
            return
        
        # Create rotation matrix around Z axis
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rot_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        
        # Rotate vertices
        rotated_vertices = mesh.vertices @ rot_matrix.T
        
        # Project to 2D (orthographic projection, just drop Z)
        # Add some perspective by scaling based on Z
        projected_vertices = []
        for v in rotated_vertices:
            # Simple perspective: scale by distance from camera
            z_offset = v[2] * 0.1  # Small perspective effect
            scale = 1.0 + z_offset
            x_2d = int(position[0] + v[0] * scale)
            y_2d = int(position[1] + v[1] * scale)
            z_depth = v[2]  # Store for depth sorting
            projected_vertices.append((x_2d, y_2d, z_depth))
        
        # Get faces and calculate depths
        faces_with_depth = []
        for face in mesh.faces:
            # Calculate average depth of face
            avg_depth = sum(projected_vertices[i][2] for i in face)
            faces_with_depth.append((avg_depth, face))
        
        # Sort faces by depth (back to front for Painter's algorithm)
        faces_with_depth.sort(key=lambda x: x[0], reverse=True)
        
        # Draw faces
        for depth, face in faces_with_depth:
            # Get 2D points for this face
            pts_2d = np.array([
                [projected_vertices[i][0], projected_vertices[i][1]] 
                for i in face
            ], np.int32)
            
            # Calculate color based on depth (lighter = closer)
            depth_factor = (depth + 2.0) / 4.0  # Normalize to 0-1 range
            depth_factor = max(0.3, min(1.0, depth_factor))  # Clamp
            
            # Use a warm color (yellow/orange) for the object
            color = (
                int(100 * depth_factor),
                int(200 * depth_factor),
                int(255 * depth_factor)
            )
            
            # Draw filled polygon
            cv2.fillPoly(frame, [pts_2d], color)
            # Draw outline
            cv2.polylines(frame, [pts_2d], True, (255, 255, 255), 1, cv2.LINE_AA)
    
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

