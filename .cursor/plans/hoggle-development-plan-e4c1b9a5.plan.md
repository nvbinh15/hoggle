<!-- e4c1b9a5-b4c6-4c0f-aee0-c7c2ab5b76db 81250f00-d731-4d88-9379-67d0707091e9 -->
# Hoggle Development Plan

## 1. Setup & Dependencies

- Create `requirements.txt` with:
    - `PyQt6` (GUI)
    - `opencv-python` (Vision)
    - `mediapipe` (Hand Tracking)
    - `google-genai==1.52.0` (AI/Voice Verification)
    - `pyaudio` (Sound Recording)
    - `gTTS` (Text-to-Speech for Hermione's voice)
    - `playsound` (Audio playback)
    - `numpy`
    - `Pillow`
- Setup basic project structure: `hoggle/` root with `gui/`, `core/`, `assets/`.

## 2. Core Modules (`core/`)

- **Hand Tracking (`hand_tracking.py`)**:
    - Initialize MediaPipe Hands.
    - Function to process frames and return landmarks.
    - Helper to calculate "Wand" position (anchored to hand).
- **Voice Verification & Feedback (`voice_verifier.py`)**:
    - Record audio from microphone.
    - Send audio/prompt to Gemini API (`google-genai`).
    - **System Instruction**: "You are Hermione Granger. Verify the user's spell pronunciation. Be strict but helpful. If wrong, correct them like 'It's Levi-O-sa, not Levio-SA'."
    - **TTS Output**: Convert Gemini's text response to speech (using `gTTS`) to play back to the user.
- **Spell Engine (`spell_engine.py`)**:
    - Manage state of current spell (active/inactive).
    - Calculate visual effects (overlays) based on spell type and wand position.
    - Effects:
        - `Lumos`: Glowing circle at wand tip.
        - `Accio`: Image sprite moving from edge to wand.
        - `Wingardium`: Image sprite hovering/bobbing above wand.

## 3. GUI Implementation (`gui/`)

- **Camera Widget (`camera_widget.py`)**:
    - Subclass `QLabel` or `QWidget`.
    - Run OpenCV capture in a `QThread` to prevent UI freeze.
    - Apply "AR" overlays (Wand image + Spell effects) on the video frame before displaying.
- **Main Window (`main_window.py`)**:
    - **Spell Selection Screen**: Buttons for the 3 spells.
    - **Spell Detail/Practice Screen**:
        - Text description of the spell.
        - "Start Practice" button.
        - Live Camera Feed.
        - "Cast Spell" button (Hold to record).
        - Feedback Label (shows Hermione's text).
        - Audio playback of Hermione's voice.

## 4. Integration & Assets

- Create/Generate simple placeholder assets:
    - `wand.png` (stick graphic).
    - `sparkle.png` (for Lumos).
    - `feather.png` (for Levitation).
    - `book.png` (for Accio).
- Wire up the "Cast" button to:

    1. Record audio.
    2. Pause AR updates (optional, or keep running).
    3. Send to Gemini (Hermione Persona).
    4. Update UI with text.
    5. Play audio response.
    6. If success, trigger `spell_engine` to show the effect.

## 5. Testing

- Verify Camera access.
- Verify Hand Tracking stability.
- Verify Audio recording.
- Verify Gemini API response with Hermione persona.
- Verify TTS playback.

### To-dos

- [ ] Create project structure and requirements.txt
- [ ] Implement Hand Tracking with MediaPipe
- [ ] Implement basic Camera Widget in PyQt6
- [ ] Implement Voice Verifier with Google GenAI
- [ ] Implement Spell Engine (AR Effects)
- [ ] Assemble Main Window and UI Flow