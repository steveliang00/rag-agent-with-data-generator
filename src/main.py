#!/usr/bin/env python3
"""
Main entry point for the Grammar Guide Bot
"""

import os
import warnings
# Disable tokenizers parallelism to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress PyPDF warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")

if __name__ == "__main__":
    from agent import Agent
    
    agent = Agent()
    print("Grammar Guide Bot started! Ask me anything about the grammar guide. \nType 'quit', 'exit', or 'q' to stop.")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            response = agent.run_agent(user_input)
            print(response)

        except Exception as e:
            print(f"Error: {e}")
            continue
