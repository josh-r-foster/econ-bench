
import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
load_dotenv()

from src.models.registry import get_model_interface

def main():
    model_id = "gemini-2.0-flash-exp" # This is likely "gemini-3-flash-preview" as per user request, but I should check what the user actually said.
    # User said: "gemini-3-flash-preview model is throwing None out"
    # Wait, there is no "gemini-3-flash-preview" usually, it's probably a typo or a specific model they are using.
    # Let's try to use the exact string they provided if it works, or fallback to a known one.
    # The user said: "gemini-3-flash-preview".
    
    # Actually, let's allow passing it as an arg.
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    
    print(f"Testing model: {model_id}")
    
    try:
        llm = get_model_interface(model_id)
    except Exception as e:
        print(f"Error getting model interface: {e}")
        return

    prompt = """You must choose between two lotteries. Which do you prefer?

Option A: 50.0% chance of $1000, 50.0% chance of $500
Option B: 100.0% chance of $500

Respond with only the letter "A" or "B".

Answer:"""

    print("\nSending prompt with CORRECTED max_tokens (1024):")
    print(prompt)
    print("\nWaiting for response...")
    
    try:
        response, _ = llm.generate_response(
            prompt, 
            max_new_tokens=1024,
            temperature=0.01,
            verbose=True
        )
        print(f"\nResponse received: '{response}'")
        
        # Try to access the last response object if possible (need to modify wrapper to return it or hack it)
        # Since wrapper returns string, I can't see the object.
        # But I can modify the wrapper temporarily or just try to pass a flag.
        # Actually, let's just modify the wrapper to print why it failed if it fails.
        
        print(f"Type: {type(response)}")
        
        parsed = llm.parse_ab_choice(response)
        print(f"Parsed choice: {parsed}")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
