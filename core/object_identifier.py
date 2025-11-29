"""
Object identifier module using Google GenAI (Gemini) for pattern recognition.
Identifies drawn patterns and returns object names.
"""
import base64
import io
import os
import tempfile
from typing import Optional
import cv2
import numpy as np
try:
    from google import genai
except ImportError:
    import genai


class ObjectIdentifier:
    """Identifies drawn patterns using Gemini AI vision."""
    
    # Valid object names that correspond to .glb files
    VALID_OBJECTS = ["ball", "cat", "heart", "pizza", "star", "wand"]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize object identifier.
        
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
    
    def identify_pattern(self, image_data: bytes) -> str:
        """
        Identify a drawn pattern from image data and return object name.
        
        Args:
            image_data: Image bytes (PNG or JPEG format)
            
        Returns:
            Object name string: one of "ball", "cat", "heart", "pizza", "star", "wand"
            Returns "wand" as default if identification fails
        """
        # Create system instruction
        system_instruction = (
            "You are analyzing a hand-drawn pattern or shape. "
            "The user has drawn something with their wand. "
            "Identify what object this pattern represents. "
            "You must respond with ONLY one of these exact words: ball, cat, heart, pizza, star, wand. "
            "If the pattern doesn't clearly match any of these, respond with 'wand' as the default. "
            "Do not include any explanation, just the single word."
        )
        
        # Create prompt
        prompt = (
            "What object does this drawn pattern represent? "
            "Respond with only one word: ball, cat, heart, pizza, star, or wand."
        )
        
        try:
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(image_data)
                tmp_file_path = tmp_file.name
            
            try:
                full_prompt = f"{system_instruction}\n\n{prompt}"
                
                # Approach 1: Try Client-based API with file upload
                if self.client and hasattr(self.client, 'models'):
                    try:
                        model = self.client.models.get("gemini-2.5-flash")
                        if hasattr(self.client, 'files'):
                            # Upload image file
                            image_file = self.client.files.upload(path=tmp_file_path)
                            response = model.generate_content(
                                contents=[
                                    {"text": full_prompt},
                                    {"file_data": {"file_uri": image_file.uri, "mime_type": "image/png"}}
                                ]
                            )
                        else:
                            # Fallback: encode image as base64
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            response = model.generate_content(
                                contents=[
                                    {"text": full_prompt},
                                    {"inline_data": {"mime_type": "image/png", "data": image_base64}}
                                ]
                            )
                    except Exception as e:
                        print(f"Client API approach failed: {e}")
                        # Try direct generate_content
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        response = self.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[
                                {"text": full_prompt},
                                {"inline_data": {"mime_type": "image/png", "data": image_base64}}
                            ]
                        )
                # Approach 2: Try genai module directly
                elif hasattr(genai, 'GenerativeModel'):
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    # Encode image as base64
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    response = model.generate_content(
                        [
                            {"text": full_prompt},
                            {"mime_type": "image/png", "data": image_base64}
                        ]
                    )
                else:
                    # Fallback: Use requests to call API directly
                    import requests
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": full_prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": image_base64
                                    }
                                }
                            ]
                        }]
                    }
                    response_data = requests.post(url, json=payload).json()
                    if 'candidates' in response_data and len(response_data['candidates']) > 0:
                        result_text = response_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        result_text = "wand"
                    
                    # Validate and return
                    result_text = result_text.strip().lower()
                    if result_text in self.VALID_OBJECTS:
                        return result_text
                    return "wand"
                
                # Extract result text from response
                if hasattr(response, 'text'):
                    result_text = response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    result_text = response.candidates[0].content.parts[0].text
                elif isinstance(response, dict) and 'candidates' in response:
                    result_text = response['candidates'][0]['content']['parts'][0]['text']
                else:
                    result_text = str(response)
                
                # Clean and validate result
                result_text = result_text.strip().lower()
                # Extract word if response contains multiple words
                words = result_text.split()
                for word in words:
                    if word in self.VALID_OBJECTS:
                        return word
                
                # Default to wand if no valid object found
                return "wand"
                
            except Exception as api_error:
                print(f"API Error in identify_pattern: {str(api_error)}")
                return "wand"
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            error_msg = f"Error identifying pattern: {str(e)}"
            print(error_msg)
            return "wand"
    
    def identify_from_canvas(self, canvas_image: np.ndarray) -> str:
        """
        Identify pattern from an OpenCV image (numpy array).
        
        Args:
            canvas_image: OpenCV BGR image (numpy array)
            
        Returns:
            Object name string
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', rgb_image)
        if not success:
            return "wand"
        
        image_bytes = buffer.tobytes()
        return self.identify_pattern(image_bytes)

