import os
import time
from typing import Optional, Dict, Tuple

from src.models.inference_controls import google_thinking_control

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from ..logger import log_model_call
except ImportError:
    def log_model_call(**_): pass

class LLMInterface:
    def __init__(self, model_id: str):
        self.model_id = model_id
        
        if genai is None:
            raise ImportError("google-genai python package is not installed. Please install it with `pip install google-genai`.")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY environment variable not set.")
            
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        print(f"Initialized Gemini interface for: {self.model_id}")

    def generate_response(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.5,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        
        if verbose:
            print("\n" + "─"*70)
            print("PROMPT:")
            print("─"*70)
            print(prompt)
            print()

        t0 = time.time()
        prompt_tokens = completion_tokens = None

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
        
        thinking_control = google_thinking_control(self.model_id)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            response_logprobs=return_logprobs, 
            logprobs=5 if return_logprobs else None,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(**thinking_control),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config
            )
            if response.text:
                content = response.text
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    prompt_tokens = getattr(um, "prompt_token_count", None)
                    completion_tokens = getattr(um, "candidates_token_count", None)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            log_model_call(
                model=self.model_id, prompt_chars=len(prompt), response="",
                latency_ms=(time.time() - t0) * 1000, valid=False,
                extra={"error_type": type(e).__name__, "error": str(e)},
            )
            raise


        log_model_call(
            model=self.model_id,
            prompt_chars=len(prompt),
            response=content,
            latency_ms=(time.time() - t0) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            valid=bool(content),
        )

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
