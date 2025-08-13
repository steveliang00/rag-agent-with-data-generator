import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

class Agent:
    """Q&A Agent to answer questions about provided documents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=openrouter_api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.5,
            timeout=30
        )

        # Create the graph
        self.app = self._create_graph()

    def _create_graph(self):
        """Create the graph for the agent"""
        workflow = StateGraph(MessagesState)

        workflow.add_edge(START, "bot")
        workflow.add_node("bot", self.bot_node)
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app

    def bot_node(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
    
    def run_agent(self, user_input: str, thread_id: str = "default"):
        initial_state  = {
            "messages" : [HumanMessage(content=user_input)]
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent, returns graph state
        response = self.app.invoke(initial_state, config=config)
        return response["messages"][-1].content

if __name__ == "__main__":
    # response = llm.invoke("How many r's are in the word 'strawberry'?")
    # print(response.content)
    agent = Agent()
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