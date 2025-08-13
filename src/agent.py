import os
from typing import List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from .state import AgentState
from .create_database import DocumentManager

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

    def _create_graph(self) -> StateGraph:
        """Create the graph for the agent"""
        workflow = StateGraph(AgentState)

        workflow.add_edge(START, "bot")
        workflow.add_node("bot", self.bot_node)
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app

    def bot_node(self, state: AgentState):
        
        messages = state["messages"] # first message initialized in run_agent
        documents = self._retrieve_docs(messages[-1].content)
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create a prompt template that includes the context
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
"""
You are a helpful assistant specialized in answering questions based on the given context.
Use the following context to provide accurate and helpful answers. 
If the context doesn't contain relevant information, politely say you don't have enough information to answer the question.

Context:
{context}
"""
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | self.llm
        
        # Invoke the LLM with the formatted prompt
        response = chain.invoke({
            "context": context,
            "messages": messages
        })
        return {"messages": [response]}
    

    def _retrieve_docs(self, user_input: str):
        """Retrieve documents from the vector store"""
        doc_manager = DocumentManager()
        db = doc_manager.load_vector_store()
        results = db.similarity_search(user_input, k=3)
        return results


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