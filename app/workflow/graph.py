from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from workflow.state import AgentState
from workflow.nodes import (
    query_rewriter_node,
    search_and_index_node,
    synthesis_node,
)

# ==============================================================================
# Linear Pipeline Graph
#
#   [START]
#      │
#      ▼
#   query_rewriter_node     (LLM) — resolves pronouns, self-contains the query
#      │
#      ▼
#   search_and_index_node   (no LLM) — search → scrape → FAISS index → retrieve
#      │
#      ▼
#   synthesis_node          (LLM) — clean prompt: query + context → final answer
#      │
#      ▼
#    [END]
#
# No conditional edges. No loops. Fully deterministic execution order.
# MemorySaver persists message history + FAISS store across session turns.
# ==============================================================================

workflow = StateGraph(AgentState)

workflow.add_node("query_rewriter",     query_rewriter_node)
workflow.add_node("search_and_index",   search_and_index_node)
workflow.add_node("synthesizer",        synthesis_node)

workflow.set_entry_point("query_rewriter")
workflow.add_edge("query_rewriter",   "search_and_index")
workflow.add_edge("search_and_index", "synthesizer")
workflow.add_edge("synthesizer",      END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)