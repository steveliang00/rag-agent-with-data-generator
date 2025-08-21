import os
from typing import List, Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from .state import AgentState
from .create_database import DocumentManager

# Load environment variables
load_dotenv()


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
openrouter_model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
mistral_model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
class Agent:
    """Q&A Agent to answer questions about provided documents"""
    
    def __init__(self, api_provider:Literal["mistral", "openrouter"] = "mistral"):
        
        if api_provider == "mistral":
            self.llm = ChatMistralAI(
                api_key=os.getenv("MISTRAL_API_KEY"),
                model=mistral_model_name,
                temperature=0.5,
                timeout=30
            )
        elif api_provider == "openrouter":
            self.llm = ChatOpenAI(
                api_key=openrouter_api_key,
                base_url=base_url,
                model=openrouter_model_name,
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
        
        # Only retrieve documents if not already cached in state
        if not state.get("documents"):
            documents = self._retrieve_docs(messages[-1].content)
            # Store documents in state for future use
            state["documents"] = documents
        else:
            # Use cached documents from previous calls
            documents = state["documents"]
            
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create a prompt template that includes the context
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
"""
You are a helpful assistant specialized in answering questions based on the retrieved document text.
Use the following document text to provide accurate and helpful answers. 
If the document text doesn't contain relevant information, politely say you don't have enough information to answer the question.
If user message is not related to the document text, respond normally to the best of your ability.

Document Text:
{document_text}
"""
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | self.llm
        
        # Invoke the LLM with the formatted prompt
        response = chain.invoke({
            "document_text": context,
            "messages": messages
        })
        
        # Return both the response and documents (to persist documents in state)
        return {
            "messages": [response],
            "documents": documents
        }
    

    def _retrieve_docs(self, user_input: str):
        """Retrieve documents from the vector store"""
        doc_manager = DocumentManager()
        
        # Check if chroma folder exists and has content
        if not self._is_chroma_populated(doc_manager):
            print("Vector store is empty. Loading and processing documents...")
            # Load documents from data folder
            docs = doc_manager.load_docs()
            # Split documents into chunks
            chunks = doc_manager.split_text(docs)
            # Embed and store the chunks
            doc_manager.embed_and_store_docs(chunks)
            print("Documents processed and stored successfully!")
        
        # Load the vector store and perform similarity search
        db = doc_manager.load_vector_store()
        results = db.similarity_search(user_input, k=3)
        return results
    
    def _is_chroma_populated(self, doc_manager: DocumentManager) -> bool:
        """Check if the chroma vector store exists and has content"""
        # Check if chroma directory exists
        if not os.path.exists(doc_manager.CHROMA_PATH):
            return False
        
        # Check if directory has any files (not just empty folder)
        if not any(os.listdir(doc_manager.CHROMA_PATH)):
            return False
            
        try:
            # Try to load the vector store and check if it has documents
            db = doc_manager.load_vector_store()
            count = db._collection.count()
            return count > 0
        except Exception:
            # If we can't load it, treat it as empty
            return False


    def run_agent(self, user_input: str, thread_id: str = "default"):
        """Run the agent with memory persistence via LangGraph checkpointer"""
        
        # Create input with just the new message - LangGraph will handle adding it to conversation history
        input_data = {
            "messages": [HumanMessage(content=user_input)]
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent - LangGraph memory will maintain conversation context
        response = self.app.invoke(input_data, config=config)
        return response["messages"][-1].content

