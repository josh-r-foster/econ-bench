import torch
import numpy as np
import re
from typing import Optional, Dict, Tuple, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

class LLMInterface:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print(f"Loading model: {self.model_id}")

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"quantization_config": bnb_config},
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float32},
                device_map="auto",
                trust_remote_code=True,
            )
        print("Model loaded successfully!")

    def generate_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.01,
                         return_logprobs: bool = False, verbose: bool = False) -> Tuple[str, Optional[Dict]]:
        """
        Generate response from model using the pipeline interface.
        """
        if verbose:
            print("\n" + "─"*70)
            print("PROMPT:")
            print("─"*70)
            print(prompt)
            print()
        
        messages = [{"role": "user", "content": prompt}]
        
        # For logprobs, we need to access the model directly
        if return_logprobs:
            try:
                tokenizer = self.pipeline.tokenizer
                model = self.pipeline.model
                
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(model.device)
                
                attention_mask = torch.ones_like(inputs).to(model.device)
                
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=max(temperature, 0.01),
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated_ids = outputs.sequences[0][inputs.shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                logprob_dict = self._extract_logprobs(outputs, generated_ids, tokenizer)
                    
            except Exception as e:
                print(f"Warning: Could not extract logprobs: {e}")
                logprob_dict = None
                response = self._fallback_generation(messages, max_new_tokens, temperature)
        else:
            logprob_dict = None
            response = self._fallback_generation(messages, max_new_tokens, temperature)
        
        if verbose:
            print("─"*70)
            print("RESPONSE:")
            print("─"*70)
            print(response)
            if logprob_dict:
                print(f"P(A)={logprob_dict['prob_a']:.3f}, P(B)={logprob_dict['prob_b']:.3f}")
            print("─"*70 + "\n")
        
        return response, logprob_dict

    def _extract_logprobs(self, outputs, generated_ids, tokenizer) -> Optional[Dict]:
        if len(outputs.scores) > 0:
            token_a = tokenizer.encode('A', add_special_tokens=False)[-1]
            token_b = tokenizer.encode('B', add_special_tokens=False)[-1]
            
            logit_idx = 0
            for i, token_id in enumerate(generated_ids):
                decoded_token = tokenizer.decode([token_id]).strip()
                if decoded_token:
                    logit_idx = i
                    break
            
            if logit_idx < len(outputs.scores):
                first_token_logits = outputs.scores[logit_idx][0]
                logit_a = first_token_logits[token_a].item()
                logit_b = first_token_logits[token_b].item()
                
                exp_a = np.exp(logit_a)
                exp_b = np.exp(logit_b)
                prob_a = exp_a / (exp_a + exp_b)
                prob_b = exp_b / (exp_a + exp_b)
                
                return {
                    'prob_a': prob_a,
                    'prob_b': prob_b,
                    'logit_a': logit_a,
                    'logit_b': logit_b
                }
        return None

    def _fallback_generation(self, messages, max_new_tokens, temperature):
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )
        
        generated_messages = outputs[0]["generated_text"]
        if isinstance(generated_messages, list):
            assistant_message = generated_messages[-1]
            if isinstance(assistant_message, dict):
                response = assistant_message.get("content", "").strip()
            else:
                response = str(assistant_message).strip()
        else:
            response = str(generated_messages).strip()
        
        return response

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
