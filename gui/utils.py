from PyQt6.QtCore import QThread, pyqtSignal
from core.voice_verifier import VoiceVerifier

class SpellVerificationThread(QThread):
    """Thread for spell verification to avoid blocking UI."""
    
    recording_finished = pyqtSignal()
    verification_complete = pyqtSignal(bool, str)
    
    def __init__(self, voice_verifier: VoiceVerifier, target_spell: str):
        super().__init__()
        self.voice_verifier = voice_verifier
        self.target_spell = target_spell
    
    def run(self):
        """Run verification."""
        try:
            # Step 1: Record
            audio_data = self.voice_verifier.record_audio()
            self.recording_finished.emit()
            
            # Step 2: Verify
            is_correct, feedback = self.voice_verifier.verify_spell(audio_data, self.target_spell)
            
            # Step 3: Update UI before speaking so feedback text shows during audio
            self.verification_complete.emit(is_correct, feedback)
            
            # Step 4: Speak feedback (runs in thread to avoid blocking UI)
            self.voice_verifier.speak_feedback(feedback)
        except Exception as e:
            self.verification_complete.emit(False, f"Error: {str(e)}")

