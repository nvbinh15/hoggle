<!-- d776b4f1-3af4-4a86-a509-6affbcf75391 11ef89e8-126e-4958-94ca-ab888e55efba -->
# Cursor Spell Implementation Plan

## 1. Dependencies

- Add `trimesh` and `scipy` to `requirements.txt` for 3D model loading and manipulation.

## 2. Core Logic (`core/`)

- **Create `core/object_identifier.py`**:
    - Implement `ObjectIdentifier` class using `google-genai`.
    - Method `identify_pattern(image_data: bytes) -> str` that sends the drawn pattern to Gemini and returns one of the predefined object names.
- **Modify `core/spell_engine.py`**:
    - Import `trimesh`.
    - Add `CURSOR` to `SpellType` enum.
    - Add state variables: `cursor_path` (list of points), `recording_start_time`, `identification_pending`.
    - Implement `activate_spell` for `CURSOR` to start recording.
    - Implement `update` to:
        - Record wand tip positions during the first 5 seconds.
        - Trigger identification phase after 5 seconds.
        - Update 3D model rotation/movement during display phase.
    - Implement `draw_effects` to:
        - Draw the path (polyline) during recording.
        - Render the 3D model during the display phase.
    - Implement a **Software Renderer** method:
        - Load `.glb` files using `trimesh`.
        - Project 3D vertices to 2D screen space.
        - Sort faces by depth (Painter's algorithm).
        - Draw faces using `cv2.fillPoly`.

## 3. GUI Integration (`gui/`)

- **Modify `gui/camera_widget.py`**:
    - Initialize `ObjectIdentifier` (or handle it via thread).
    - In `update_frame`, check if `spell_engine` requests identification.
    - If identification is needed:
        - Extract the drawn pattern as an image.
        - Launch a background thread to call `ObjectIdentifier`.
    - When identification completes, update `spell_engine` with the object name to load.
- **Modify `gui/main_window.py`**:
    - Add "Cursor" to the spell list in `SpellSelectionScreen`.
    - Update `PracticeScreen` to handle the specific feedback for the Cursor spell.

## 4. Assets

- Ensure `.glb` files in `assets/3d/` are loadable (User provided).

## 5. Verification

- Test "Cursor" voice trigger.
- Test drawing recording and annotation.
- Test Gemini identification.
- Test 3D model rendering and animation.

### To-dos

- [ ] Add dependencies to requirements.txt
- [ ] Create core/object_identifier.py
- [ ] Update core/spell_engine.py with CURSOR logic and 3D rendering
- [ ] Update gui/camera_widget.py to handle identification flow
- [ ] Update gui/main_window.py to add Cursor spell UI