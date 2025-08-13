#!/usr/bin/env python3
"""
Main entry point for the Grammar Guide Bot
"""

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
