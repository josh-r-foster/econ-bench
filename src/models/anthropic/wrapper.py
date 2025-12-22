import os
import re
from typing import Optional, Dict, Tuple
import numpy as np

try:
    import anthropic
except ImportError:
    anthropic = None

from dotenv import load_dotenv
load_dotenv()

class LLMInterface:
    def __init__(self, model_id: str = "claude-3-5-sonnet-20240620", device: str = "auto"):
        self.model_id = model_id
        if anthropic is None:
            raise ImportError("Anthropic python package is not installed. Please install it with `pip install anthropic`.")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY environment variable not set.")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"Initialized Anthropic interface for: {self.model_id}")

    def generate_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.01,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        
        if verbose:
            print("\n" + "─"*70)
            print("PROMPT:")
            print("─"*70)
            print(prompt)
            print()

        try:
            # Anthropic uses max_tokens, not max_new_tokens
            kwargs = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }

            # Note: Anthropic API doesn't support logprobs in the same way OpenAI does for public release yet
            # or it requires specific beta headers/flags if available. 
            # For now, we will ignore return_logprobs for Anthropic or handle it if it becomes standard.
            # If logprobs are strictly required, this might be a limitation.

            response = self.client.messages.create(**kwargs)
            
            # Extract content from TextBlock
            content = ""
            if response.content and response.content[0].type == 'text':
                content = response.content[0].text.strip()
            
            logprob_dict = None
            # Logprobs extraction not implemented for Anthropic in this basic wrapper
            if return_logprobs:
                if verbose:
                    print("Note: Logprobs not supported for Anthropic models in this wrapper.")

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            content = ""
            logprob_dict = None

        if verbose:
            print("─"*70)
            print("RESPONSE:")
            print("─"*70)
            print(content)
            print("─"*70 + "\n")

        return content, logprob_dict

    def parse_ab_choice(self, response: str) -> Optional[str]:
        if 'Answer:' in response:
            parts = response.split('Answer:')
            if len(parts) > 1:
                response = parts[-1].strip()
        
        response_clean = response.strip().upper()
        
        if response_clean.startswith("A"):
            return "A"
        if response_clean.startswith("B"):
            return "B"
        
        # Regex fallback
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
