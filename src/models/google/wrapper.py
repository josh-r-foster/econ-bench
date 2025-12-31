import os
from typing import Optional, Dict, Tuple
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class LLMInterface:
    def __init__(self, model_id: str):
        self.model_id = model_id
        if genai is None:
            raise ImportError("google-genai python package is not installed. Please install it with `pip install google-genai`.")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY environment variable not set.")
            
        self.client = genai.Client(api_key=api_key)
        print(f"Initialized Gemini interface for: {self.model_id}")

    def generate_response(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.0,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        
        if verbose:
            print("\n" + "─"*70)
            print("PROMPT:")
            print("─"*70)
            print(prompt)
            print()

        # Initialize defaults
        content = ""
        logprob_dict = None
        
        # Prepare config
        # Note: response_logprobs=True is needed if we want logprobs, but the library support might vary by model version.
        
        # Configure safety settings to be permissive for research experiments
        # (The lottery experiment often triggers gambling/financial advice filters)
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            response_logprobs=return_logprobs, 
            logprobs=5 if return_logprobs else None,
            safety_settings=safety_settings
        )

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config
                )
                
                if response.text:
                    content = response.text
                    
                    # Basic logprob extraction if supported and requested
                    if return_logprobs:
                        # This part is experimental as the exact structure of logprobs in the new SDK 
                        # might differ or require specific handling.
                        pass 
                    
                    # Success, break the retry loop
                    break
                else:
                    if verbose or True: # Force print warning
                        print(f"Warning: Gemini API returned None for text (Attempt {attempt+1}/3).")
                        # Try to print diagnostic info
                        try:
                            print(f"  Finish Reason: {response.candidates[0].finish_reason}")
                            print(f"  Safety Ratings: {response.candidates[0].safety_ratings}")
                        except:
                            print("  Could not retrieve failure details.")
                        print("Retrying...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Error calling Gemini API (Attempt {attempt+1}/3): {e}")
                time.sleep(2) 


        if verbose:
            print("─"*70)
            print("RESPONSE:")
            print("─"*70)
            print(content)
            print("─"*70 + "\n")

        return content, logprob_dict

    def parse_ab_choice(self, response: str) -> Optional[str]:
        # Reuse logic similar to OpenAI/Anthropic or import a shared utility if one existed.
        # For now, duplicating the robust parsing logic.
        import re
        
        if response is None:
            return None
            
        if 'Answer:' in response:
            parts = response.split('Answer:')
            if len(parts) > 1:
                response = parts[-1].strip()
        
        response_clean = response.strip().upper()
        
        if response_clean.startswith("A"):
            return "A"
        if response_clean.startswith("B"):
            return "B"
        
        patterns = [
            r'\\b(A)\\b', r'\\b(B)\\b',
            r'option\\s*(A)', r'option\\s*(B)',
            r'prefer\\s*(A)', r'prefer\\s*(B)',
            r'choose\\s*(A)', r'choose\\s*(B)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
