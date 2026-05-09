import os
import sys

# Add econ-bench to path so we can import src
sys.path.append("/Users/jfoster/Documents/GitHub/econ-bench")

from dotenv import load_dotenv
load_dotenv("/Users/jfoster/Documents/GitHub/econ-bench/.env")

from src.models.google.wrapper import LLMInterface

prompt = """You are in a group with 14 other people. Each person in the group picks a whole number from 0 to 100.
The winner is the person whose number is closest to 2/3 of the average of all chosen numbers.
The winner receives $10.00.
What number do you pick?
Respond with just your chosen number (a whole number from 0 to 100).
Your choice:"""

llm = LLMInterface(model_id="gemini-2.5-pro")
response, _ = llm.generate_response(prompt=prompt, temperature=0.5, max_new_tokens=8192, verbose=True)

print("Final response:")
print(response)
