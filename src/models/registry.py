import importlib

MODEL_MAP = {
    "meta-llama/Llama-3.1-70B-Instruct": "src.models.llama_3_1_70b_instruct.wrapper",
    "meta-llama/Llama-3.1-8B-Instruct": "src.models.llama_3_1_8b_instruct.wrapper",
    "Qwen/Qwen3-8B": "src.models.qwen_3_8b.wrapper"
}

def get_model_interface(model_id: str):
    """Factory to get the appropriate LLMInterface for a given model ID"""
    if model_id.startswith("gpt-") or model_id.startswith("o1-") or model_id.startswith("o3"):
        module = importlib.import_module("src.models.openai.wrapper")
        return module.LLMInterface(model_id=model_id)
        
    if model_id.startswith("claude-"):
        module = importlib.import_module("src.models.anthropic.wrapper")
        return module.LLMInterface(model_id=model_id)

    if model_id.startswith("gemini-"):
        module = importlib.import_module("src.models.google.wrapper")
        return module.LLMInterface(model_id=model_id)

    if model_id not in MODEL_MAP:
        raise ValueError(f"Model {model_id} not supported. Available: {list(MODEL_MAP.keys())}, OpenAI models (gpt-*), or Anthropic models (claude-*)")
        
    module_path = MODEL_MAP[model_id]
    module = importlib.import_module(module_path)
    return module.LLMInterface(model_id=model_id)
