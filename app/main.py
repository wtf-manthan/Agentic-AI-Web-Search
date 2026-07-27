from workflow.graph import app
from langchain_core.messages import HumanMessage
import uuid

SESSION_THREAD_ID = str(uuid.uuid4())
SESSION_CONFIG    = {"configurable": {"thread_id": SESSION_THREAD_ID}}


def ask_question(question: str) -> str:
    """
    Submit a question into the linear pipeline under the active session.

    MemorySaver restores message history across turns automatically.
    Per-turn fields (rewritten_query, rag_results, final_answer) are
    reset each call so every question goes through a fresh search cycle.
    The 'messages' list accumulates across turns via the add_messages reducer,
    giving the query rewriter its conversation context window.
    """
    state = {
        "messages":        [HumanMessage(content=question)],
        "original_query":  question,
        "rewritten_query": "",
        "rag_results":     "",
        "final_answer":    "",
    }
    result = app.invoke(state, config=SESSION_CONFIG)
    return result["final_answer"]


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Research Agent  (Query Rewriter + Playwright + FAISS)")
    print("  Type 'exit' or 'quit' to end the session.")
    print(f"  Session ID: {SESSION_THREAD_ID}")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        answer = ask_question(query)
        print(f"\nAgent: {answer}\n")