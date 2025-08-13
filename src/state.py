from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """State for the data analyst agent."""
    
    # Message history
    messages: Annotated[List[BaseMessage], add_messages]

    # Documents
    documents: Optional[List[Document]]