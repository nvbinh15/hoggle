"""
Voice verification module using Google GenAI (Gemini) for spell pronunciation checking.
Hermione Granger persona provides feedback.
"""
import io
import base64
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import subprocess
import platform
from typing import Optional, Tuple
try:
    from google import genai
except ImportError:
    import genai
from gtts import gTTS

try:
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False


class VoiceVerifier:
    """Records audio and verifies spell pronunciation with Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize voice verifier.
        
        Args:
            api_key: Google GenAI API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key.")
        
        # Initialize client - try different API structures
        try:
            self.client = genai.Client(api_key=self.api_key)
        except:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai
            except:
                # Fallback: store API key for direct API calls
                self.client = None
                self.api_key = self.api_key

        # Initialize ElevenLabs
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.elevenlabs_client = None
        self.elevenlabs_voice_id = "XB0fDUnXU5powFXDhCwa" # Default to Charlotte
        
        if HAS_ELEVENLABS and self.elevenlabs_api_key:
            try:
                self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
                print("ElevenLabs initialized successfully.")
            except Exception as e:
                print(f"Failed to init ElevenLabs: {e}")
        
        # Audio recording settings
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 3  # Record for 3 seconds
        self.dtype = np.int16  # 16-bit audio
    
    def record_audio(self) -> bytes:
        """
        Record audio from microphone.
        
        Returns:
            Audio data as bytes (WAV format)
        """
        print("Recording... Speak now!")
        
        # Record audio using sounddevice
        audio_data = sd.rec(
            int(self.rate * self.record_seconds),
            samplerate=self.rate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()  # Wait until recording is finished
        
        print("Finished recording.")
        
        # Convert numpy array to bytes
        audio_bytes = audio_data.tobytes()
        
        # Convert to WAV bytes format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
            wf.setframerate(self.rate)
            wf.writeframes(audio_bytes)
        
        return wav_buffer.getvalue()
    
    def verify_spell(self, audio_data: bytes, target_spell: str) -> Tuple[bool, str]:
        """
        Verify spell pronunciation using Gemini AI with Hermione persona.
        
        Args:
            audio_data: WAV audio bytes
            target_spell: The spell name to check (e.g., "Lumos", "Wingardium Leviosa")
            
        Returns:
            Tuple of (is_correct: bool, feedback_message: str)
        """
        # Convert audio to base64 for Gemini
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Create system instruction for Hermione persona
        system_instruction = (
            "You are Hermione Granger from Harry Potter. You are teaching a muggle how to cast spells. "
            "The user will attempt to pronounce a spell. Your job is to verify if they pronounced it correctly. "
            "Be strict but helpful. If they got it wrong, correct them in Hermione's characteristic way. "
            "For example, if they say 'Wingardium Levio-SA' instead of 'Wingardium Levi-O-sa', "
            "you should say: 'It's Levi-O-sa, not Levio-SA!' "
            "If they got it right, praise them enthusiastically. "
            "Keep your response brief and in character."
        )
        
        # Create prompt
        prompt = (
            f"The user is trying to cast the spell '{target_spell}'. "
            f"Listen to their pronunciation and tell me if they got it right. "
            f"Provide feedback as Hermione Granger."
        )
        
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Try multiple API approaches for google-genai 1.52.0
                full_prompt = f"{system_instruction}\n\n{prompt}"
                
                # Approach 1: Try Client-based API
                if self.client and hasattr(self.client, 'models'):
                    try:
                        model = self.client.models.get("gemini-2.5-flash")
                        if hasattr(self.client, 'files'):
                            # Upload audio file
                            audio_file = self.client.files.upload(path=tmp_file_path)
                            response = model.generate_content(
                                contents=[
                                    {"text": full_prompt},
                                    {"file_data": {"file_uri": audio_file.uri, "mime_type": "audio/wav"}}
                                ]
                            )
                        else:
                            # Text-only fallback
                            response = model.generate_content(
                                contents=f"{full_prompt}\n\nNote: Audio was recorded but file upload not available. Please provide general feedback about pronouncing '{target_spell}'."
                            )
                    except:
                        # Try direct generate_content
                        response = self.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=full_prompt
                        )
                # Approach 2: Try genai module directly
                elif hasattr(genai, 'GenerativeModel'):
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    # For audio, we'd need to use file upload - for now use text
                    response = model.generate_content(
                        f"{full_prompt}\n\nThe user attempted to say '{target_spell}'. Please provide feedback as Hermione."
                    )
                else:
                    # Fallback: Use requests to call API directly
                    import requests
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{
                            "parts": [{"text": f"{full_prompt}\n\nThe user attempted to say '{target_spell}'. Please provide feedback as Hermione."}]
                        }]
                    }
                    response_data = requests.post(url, json=payload).json()
                    if 'candidates' in response_data and len(response_data['candidates']) > 0:
                        feedback = response_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        feedback = "I couldn't verify your pronunciation. Please try again."
                    feedback_lower = feedback.lower()
                    is_correct = (
                        "correct" in feedback_lower or 
                        "right" in feedback_lower or 
                        "well done" in feedback_lower or
                        "excellent" in feedback_lower
                    )
                    if "wrong" in feedback_lower or "incorrect" in feedback_lower or "not" in feedback_lower:
                        is_correct = False
                    return is_correct, feedback
                
                # Extract feedback text from response
                if hasattr(response, 'text'):
                    feedback = response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    feedback = response.candidates[0].content.parts[0].text
                elif isinstance(response, dict) and 'candidates' in response:
                    feedback = response['candidates'][0]['content']['parts'][0]['text']
                else:
                    feedback = str(response)
                
                # Determine if correct
                feedback_lower = feedback.lower()
                is_correct = (
                    "correct" in feedback_lower or 
                    "right" in feedback_lower or 
                    "well done" in feedback_lower or
                    "excellent" in feedback_lower or
                    "perfect" in feedback_lower or
                    "brilliant" in feedback_lower
                )
                if "wrong" in feedback_lower or "incorrect" in feedback_lower or ("not" in feedback_lower and "correct" not in feedback_lower):
                    is_correct = False
                
                return is_correct, feedback
                
            except Exception as api_error:
                # Final fallback - provide basic feedback
                error_msg = f"API Error: {str(api_error)}"
                print(error_msg)
                # Return a helpful message
                feedback = f"I had trouble verifying that. Please try saying '{target_spell}' more clearly."
                return False, feedback
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            error_msg = f"Error verifying spell: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def speak_feedback(self, text: str, lang: str = 'en') -> None:
        """
        Convert text to speech and play it (Hermione's voice feedback).
        Uses ElevenLabs if available, falls back to gTTS.
        
        Args:
            text: Text to speak
            lang: Language code (default: 'en')
        """
        try:
            tmp_file_path = None
            
            # Try ElevenLabs first
            if self.elevenlabs_client:
                try:
                    # Use Charlotte (British female) or default voice
                    # Using client.text_to_speech.convert() which is the standard method in v3+
                    audio_generator = self.elevenlabs_client.text_to_speech.convert(
                        text=text,
                        voice_id=self.elevenlabs_voice_id,
                        model_id="eleven_multilingual_v2"
                    )
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        for chunk in audio_generator:
                            tmp_file.write(chunk)
                        tmp_file_path = tmp_file.name
                except Exception as e:
                    print(f"ElevenLabs generation failed: {e}")
                    tmp_file_path = None

            # Fallback to gTTS
            if not tmp_file_path:
                tts = gTTS(text=text, lang=lang, slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    tmp_file_path = tmp_file.name
            
            # Play audio using system player
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.run(['afplay', tmp_file_path], check=False)
            elif system == 'Linux':
                subprocess.run(['mpg123', tmp_file_path], check=False)
            elif system == 'Windows':
                subprocess.run(['start', tmp_file_path], shell=True, check=False)
            else:
                # Fallback: try common players
                try:
                    subprocess.run(['ffplay', '-nodisp', '-autoexit', tmp_file_path], check=False)
                except:
                    print(f"Could not play audio. Please install a media player.")
            
            # Clean up
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
        except Exception as e:
            print(f"Error playing TTS: {str(e)}")
    
    def verify_and_feedback(self, target_spell: str) -> Tuple[bool, str]:
        """
        Record audio, verify spell, and provide both text and voice feedback.
        
        Args:
            target_spell: The spell to verify
            
        Returns:
            Tuple of (is_correct: bool, feedback_message: str)
        """
        audio_data = self.record_audio()
        is_correct, feedback = self.verify_spell(audio_data, target_spell)
        
        # Play voice feedback
        self.speak_feedback(feedback)
        
        return is_correct, feedback
    
    def release(self):
        """Release audio resources."""
        # sounddevice doesn't require explicit cleanup
        pass

