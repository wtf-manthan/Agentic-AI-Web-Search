from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]  # full session history
    original_query:  str   # raw user input
    rewritten_query: str   # self-contained query from Node 1
    rag_results:     str   # retrieved FAISS chunks from Node 2
    final_answer:    str   # synthesized answer from Node 3