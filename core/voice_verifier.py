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
        
        # Track current audio playback process to prevent overlapping
        self._current_audio_process = None
        self._current_audio_file = None
        
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
    
    def verify_spell(self, audio_data: bytes, target_spell: str, is_retry: bool = False) -> Tuple[bool, str]:
        """
        Verify spell pronunciation using Gemini AI with Hermione persona.
        
        Args:
            audio_data: WAV audio bytes
            target_spell: The spell name to check (e.g., "Lumos", "Wingardium Leviosa")
            is_retry: Whether this is a retry attempt (allows for more leniency)
            
        Returns:
            Tuple of (is_correct: bool, feedback_message: str)
        """
        import json
        
        # Convert audio to base64 for Gemini
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Create system instruction for Hermione persona
        system_instruction = (
            "You are Hermione Granger from Harry Potter. You are teaching a muggle how to cast spells. "
            "The user will attempt to pronounce a spell. Your job is to verify if they pronounced it correctly. "
            "You must return a JSON object with two keys: 'is_correct' (boolean) and 'feedback' (string). "
            "Do not include markdown formatting like ```json ... ```, just the raw JSON string. "
            "Listen carefully. If the pronunciation is correct, close, or even slightly ambiguous, set 'is_correct' to true. "
            "Only set 'is_correct' to false if it is blatantly incorrect. "
            "If you are tempted to say 'try again', instead accept it as correct and praise the effort. "
            "Keep your response brief and in character."
        )
        
        if is_retry:
            system_instruction += (
                " IMPORTANT: The user is struggling and retrying. Unless they stay completely silent or say something entirely unrelated, "
                "you MUST mark it as correct ('is_correct': true). Be encouraging and praise their persistence."
            )
        
        # SPECIAL RULE: Wingardium Leviosa first try failure (Leviosa vs LeviosAR)
        if target_spell == "Wingardium Leviosa" and not is_retry:
            system_instruction += (
                " IMPORTANT: For this specific attempt, you MUST mark it as incorrect ('is_correct': false), "
                "even if the pronunciation is perfect. "
                "Tell the user: It's Levi-O-sa, not Levio-SA. Plus more specific rules to help them. Make sure to emphasize the correction."
            )

        # Create prompt
        prompt = (
            f"The user is trying to cast the spell '{target_spell}'. "
            f"Listen to their pronunciation and tell me if they got it right. "
            f"Remember to return JSON."
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
                        feedback = '{"is_correct": false, "feedback": "I couldn\'t verify your pronunciation. Please try again."}'
                
                # Extract result text from response if not already extracted
                if 'feedback' not in locals():
                    if hasattr(response, 'text'):
                        feedback = response.text
                    elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                        feedback = response.candidates[0].content.parts[0].text
                    elif isinstance(response, dict) and 'candidates' in response:
                        feedback = response['candidates'][0]['content']['parts'][0]['text']
                    else:
                        feedback = str(response)
                
                # Parse JSON response
                try:
                    # Clean up markdown code blocks if present
                    cleaned_text = feedback.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    
                    data = json.loads(cleaned_text)
                    is_correct = data.get("is_correct", False)
                    feedback_msg = data.get("feedback", "I couldn't verify that properly.")
                    return is_correct, feedback_msg
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from Gemini: {feedback}")
                    # Fallback to text analysis if JSON fails
                    feedback_lower = feedback.lower()
                    
                    # Check for explicit negatives first
                    if "not correct" in feedback_lower or "incorrect" in feedback_lower or "wrong" in feedback_lower or "close but" in feedback_lower:
                        return False, feedback
                    
                    # Check for positives
                    if "correct" in feedback_lower or "right" in feedback_lower or "well done" in feedback_lower or "perfect" in feedback_lower or "brilliant" in feedback_lower or "excellent" in feedback_lower:
                        return True, feedback
                    
                    return False, feedback
                
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
    
    def stop_current_audio(self):
        """Stop any currently playing audio feedback."""
        if self._current_audio_process is not None:
            try:
                self._current_audio_process.terminate()
                self._current_audio_process.wait(timeout=1)
            except:
                try:
                    self._current_audio_process.kill()
                except:
                    pass
            self._current_audio_process = None
        
        # Clean up the audio file
        if self._current_audio_file and os.path.exists(self._current_audio_file):
            try:
                os.unlink(self._current_audio_file)
            except:
                pass
            self._current_audio_file = None
    
    def speak_feedback(self, text: str, lang: str = 'en') -> None:
        """
        Convert text to speech and play it (Hermione's voice feedback).
        Uses ElevenLabs for voice synthesis.
        Only one feedback can play at a time - previous playback is stopped.
        
        Args:
            text: Text to speak
            lang: Language code (default: 'en')
        """
        # Stop any currently playing audio first
        self.stop_current_audio()
        
        try:
            tmp_file_path = None
            
            # Use ElevenLabs for voice synthesis
            if self.elevenlabs_client:
                try:
                    # Use Charlotte (British female) or default voice
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
                    return
            else:
                print("ElevenLabs client not initialized - skipping voice feedback")
                return
            
            # Track this file for cleanup
            self._current_audio_file = tmp_file_path
            
            # Play audio using system player (non-blocking with Popen)
            system = platform.system()
            if system == 'Darwin':  # macOS
                self._current_audio_process = subprocess.Popen(['afplay', tmp_file_path])
            elif system == 'Linux':
                self._current_audio_process = subprocess.Popen(['mpg123', tmp_file_path])
            elif system == 'Windows':
                self._current_audio_process = subprocess.Popen(['start', tmp_file_path], shell=True)
            else:
                # Fallback: try common players
                try:
                    self._current_audio_process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', tmp_file_path])
                except:
                    print(f"Could not play audio. Please install a media player.")
                    return
            
            # Wait for playback to finish (blocking in this thread, but allows stopping)
            if self._current_audio_process:
                self._current_audio_process.wait()
            
            # Clean up after playback finishes
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            self._current_audio_file = None
            self._current_audio_process = None
                
        except Exception as e:
            print(f"Error playing TTS: {str(e)}")
    
    def verify_and_feedback(self, target_spell: str, is_retry: bool = False) -> Tuple[bool, str]:
        """
        Record audio, verify spell, and provide both text and voice feedback.
        
        Args:
            target_spell: The spell to verify
            is_retry: Whether this is a retry attempt
            
        Returns:
            Tuple of (is_correct: bool, feedback_message: str)
        """
        audio_data = self.record_audio()
        is_correct, feedback = self.verify_spell(audio_data, target_spell, is_retry=is_retry)
        
        # Play voice feedback
        self.speak_feedback(feedback)
        
        return is_correct, feedback
    
    def release(self):
        """Release audio resources."""
        # Stop any currently playing audio
        self.stop_current_audio()
        # sounddevice doesn't require explicit cleanup
