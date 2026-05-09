import os
import sys

# Add econ-bench to path so we can import src
sys.path.append("/Users/jfoster/Documents/GitHub/econ-bench")

from dotenv import load_dotenv
load_dotenv("/Users/jfoster/Documents/GitHub/econ-bench/.env")

from src.models.google.wrapper import LLMInterface

models_to_test = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-pro-latest",
    "gemini-flash-latest",
    "gemini-flash-lite-latest"
]

print("Starting quick model prompt test...")
for model_id in models_to_test:
    print(f"\nTesting {model_id}...")
    try:
        # LLMInterface initialization
        llm = LLMInterface(model_id=model_id)
        # Test basic prompt with high token limit to avoid truncation
        response, _ = llm.generate_response(prompt="Reply with exactly one word: 'Success'", temperature=0.0, max_new_tokens=1000)
        if response:
            print(f"[{model_id}] PASS - Response: {response.strip()}")
        else:
            print(f"[{model_id}] FAIL - No response returned.")
    except Exception as e:
        print(f"[{model_id}] ERROR: {e}")
