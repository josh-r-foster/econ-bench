import os
from typing import Optional, Dict, Tuple
import numpy as np
import re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class LLMInterface:
    def __init__(self, model_id: str = "gpt-4o", device: str = "auto"):
        self.model_id = model_id
        if OpenAI is None:
            raise ImportError("OpenAI python package is not installed. Please install it with `pip install openai`.")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try to get from client if already configured globally (less likely but possible)
            pass
            # We don't raise strict error here to allow for some other auth methods if openai client handles it, 
            # but usually key is needed.
            print("Warning: OPENAI_API_KEY environment variable not set.")
            
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI interface for: {self.model_id}")

    def generate_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.01,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        
        if verbose:
            print("\n" + "─"*70)
            print("PROMPT:")
            print("─"*70)
            print(prompt)
            print()

        try:
            # Handle parameter differences for newer models (o1, gpt-5, etc)
            # These models require 'max_completion_tokens' instead of 'max_tokens'
            token_param = "max_completion_tokens" if self.model_id.startswith(("o1", "o3", "gpt-5")) else "max_tokens"

            # Prepare arguments
            kwargs = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                token_param: 5000 if self.model_id.startswith(("o1", "o3", "gpt-5")) else max_new_tokens,
                "temperature": 1 if self.model_id.startswith(("o1", "o3", "gpt-5")) else temperature,
            }

            if return_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5  # Capture enough to find A and B

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
            
            logprob_dict = None
            if return_logprobs and response.choices[0].logprobs:
                # Extract logprobs from the first token
                content_logprobs = response.choices[0].logprobs.content
                if content_logprobs:
                    first_token_logprobs = content_logprobs[0].top_logprobs
                    logprob_dict = self._extract_logprobs(first_token_logprobs)

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            content = ""
            logprob_dict = None

        if verbose:
            print("─"*70)
            print("RESPONSE:")
            print("─"*70)
            print(content)
            if logprob_dict:
                print(f"P(A)={logprob_dict.get('prob_a', 0):.3f}, P(B)={logprob_dict.get('prob_b', 0):.3f}")
            print("─"*70 + "\n")

        return content, logprob_dict

    def _extract_logprobs(self, top_logprobs) -> Optional[Dict]:
        # top_logprobs is a list of TopLogprob objects
        # We look for tokens that represent "A" and "B"
        
        prob_a = 0.0
        prob_b = 0.0
        logit_a = -100.0 # OpenAI doesn't return raw logits easily in this struct, but returns logprob
        logit_b = -100.0
        
        found_a = False
        found_b = False

        for item in top_logprobs:
            token = item.token.strip().upper()
            if token == "A":
                prob_a = np.exp(item.logprob)
                logit_a = item.logprob # Approximate logit as logprob for storage
                found_a = True
            elif token == "B":
                prob_b = np.exp(item.logprob)
                logit_b = item.logprob
                found_b = True
        
        # Normalize if both found, or just return what we have
        if found_a or found_b:
            total = prob_a + prob_b
            if total > 0:
                norm_a = prob_a / total
                norm_b = prob_b / total
                return {
                    'prob_a': norm_a,
                    'prob_b': norm_b,
                    'logit_a': logit_a,
                    'logit_b': logit_b
                }
        
        return None

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
