# Hoggle - Hogwarts for Muggles

Learn to cast spells with Hermione Granger as your instructor! This interactive GUI application teaches you three spells using hand tracking, voice recognition, and augmented reality.

## Features

- **Three Spells to Learn:**
  - **Lumos**: Light up the tip of your wand
  - **Accio**: Summon objects to you
  - **Wingardium Leviosa**: Make objects float

- **Interactive Practice:**
  - Real-time hand tracking using MediaPipe
  - Virtual wand overlay on your hand
  - Voice recognition for spell pronunciation
  - Hermione Granger provides text and voice feedback

- **Visual Effects:**
  - AR wand visualization
  - Spell-specific visual effects when cast correctly

## Requirements

- Python 3.8 or higher
- Webcam
- Microphone
- Google GenAI API key

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Google API key:
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Or on Windows:
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

## Usage

Run the application:
```bash
python main.py
```

1. Select a spell from the main menu
2. Click "Start Practice" to begin
3. Hold the "Cast Spell" button and speak the spell name
4. Hermione will verify your pronunciation and provide feedback
5. If correct, see the spell effect!

## How It Works

- **Hand Tracking**: Uses MediaPipe to detect your hand and track finger positions
- **Wand Visualization**: Overlays a virtual wand on your index finger
- **Voice Verification**: Records your voice and sends it to Google Gemini AI
- **Hermione Persona**: Gemini responds as Hermione Granger with helpful feedback
- **Text-to-Speech**: Converts Hermione's feedback to voice using gTTS
- **Spell Effects**: Visual effects are rendered based on the spell type

## Troubleshooting

- **Camera not working**: Make sure your webcam is connected and not being used by another application
- **Microphone not working**: Check your system audio settings and permissions
- **API errors**: Verify your GOOGLE_API_KEY is set correctly
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## Notes

- The application requires an internet connection for voice verification
- Audio recording duration is set to 3 seconds (adjustable in `core/voice_verifier.py`)
- Camera feed runs at approximately 30 FPS

## License

This project is for educational purposes.

