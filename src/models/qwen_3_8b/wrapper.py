import torch
import numpy as np
import re
from typing import Optional, Dict, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMInterface:
    def __init__(self, model_id: str = "Qwen/Qwen3-8B", device: str = "auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print(f"Loading model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        }
        
        # Check for 4bit but default purely to loaded args from snippet logic, 
        # though snippet had optional 4bit. I'll stick to standard loading to be safe
        # unless user asks for quantization optimization later.
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        print("Model loaded successfully!")

    def generate_response(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.01,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        
        if verbose:
            print(f"\\n{'─'*70}\\nPROMPT:\\n{'─'*70}\\n{prompt}\\n")

        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        attention_mask = torch.ones_like(inputs).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        input_len = inputs.shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        logprob_dict = None
        if return_logprobs:
            # Qwen logprob extraction logic similar to Llama but adapted if needed
            # For now, reuse the generic extraction logic if compatible, 
            # otherwise return None (soft MCMC will fallback to hard)
            logprob_dict = self._extract_logprobs(outputs, generated_ids)

        if verbose:
             print(f"{'─'*70}\\nRESPONSE:\\n{'─'*70}\\n{response}\\n{'─'*70}\\n")

        return response, logprob_dict

    def _extract_logprobs(self, outputs, generated_ids) -> Optional[Dict]:
        if not hasattr(outputs, 'scores') or len(outputs.scores) == 0:
            return None
            
        try:
            token_a_id = self.tokenizer.encode('A', add_special_tokens=False)[-1]
            token_b_id = self.tokenizer.encode('B', add_special_tokens=False)[-1]
            
            # Find first non-whitespace token after generation starts
            # Note: Qwen might output <think> first. 
            # If so, Soft MCMC is tricky because the choice comes LATER.
            # Strategy: Soft MCMC works best on "immediate" choices. 
            # If the model thinks first, we can't easily get A/B prob at step 1.
            # So for Qwen, we might return None logprobs if it starts with <think>.
            
            first_decoded = self.tokenizer.decode(generated_ids[0]).strip()
            if first_decoded.startswith("<") or not first_decoded:
                # Likely <think> or whitespace, skip logprobs (fallback to Hard MCMC)
                return None
                
            first_token_logits = outputs.scores[0][0]
            logit_a = first_token_logits[token_a_id].item()
            logit_b = first_token_logits[token_b_id].item()
            
            exp_a = np.exp(logit_a)
            exp_b = np.exp(logit_b)
            return {
                'prob_a': exp_a / (exp_a + exp_b),
                'prob_b': exp_b / (exp_a + exp_b)
            }
        except Exception:
            return None

    def parse_ab_choice(self, response: str) -> Optional[str]:
        # Handle <think> tags
        think_match = re.search(r'</think>(.*)', response, re.IGNORECASE | re.DOTALL)
        if think_match:
            response_to_parse = think_match.group(1).strip()
        else:
            response_to_parse = response

        response_clean = response_to_parse.strip().upper()
        if response_clean.startswith("A"): return "A"
        if response_clean.startswith("B"): return "B"
        
        patterns = [
            r'\\b(A)\\b', r'\\b(B)\\b',
            r'option\\s*(A)', r'option\\s*(B)',
            r'prefer\\s*(A)', r'prefer\\s*(B)',
        ]
        
        # Check after think block first
        for p in patterns:
            if re.search(p, response_to_parse, re.IGNORECASE):
                return re.search(p, response_to_parse, re.IGNORECASE).group(1).upper()
        
        # Fallback to full response
        if think_match:
            for p in patterns:
                if re.search(p, response, re.IGNORECASE):
                     return re.search(p, response, re.IGNORECASE).group(1).upper()
        
        return None
