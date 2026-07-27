from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage, AIMessage

from workflow.state import AgentState
from search_tools import web_search_agent
from retriever import rag_pipeline
from scraper import scrape_and_clean_url


# How many recent turns to show the rewriter for context (1 turn = 1 user + 1 agent msg)
REWRITER_CONTEXT_TURNS = 3


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

# ------------------------------------------------------------------------------
# Node 1 — Query Rewriter
#
# Role   : Resolve ambiguous references using recent conversation context
# Input  : Last N turns of conversation + the new raw user question
# Output : A single, fully self-contained question — nothing else
# ------------------------------------------------------------------------------
REWRITER_PROMPT_TEMPLATE = """\
ROLE
────
You are a Query Resolver. Your job is to rewrite an ambiguous research question \
into a fully self-contained one using recent conversation context.

RECENT CONVERSATION (last {n} turns)
──────────────────────────────────────
{recent_history}

NEW QUESTION
────────────
{current_query}

TASK
────
Rewrite the question so it is completely self-contained:
• Replace all pronouns and vague references with explicit names or entities \
from the conversation (e.g. "he" → the person's full name, "it" → the product name)
• If the question is already self-contained, return it unchanged
• Output ONLY the rewritten question — no explanation, no punctuation changes, \
no extra text

REWRITTEN QUESTION:\
"""


# ------------------------------------------------------------------------------
# Node 3 — Synthesizer
#
# Role   : Write the final answer
# Input  : Self-contained question + retrieved FAISS chunks
# Output : A clear, grounded answer — no routing language, no meta-commentary
# Note   : Message history is intentionally excluded to keep the prompt clean
# ------------------------------------------------------------------------------
SYNTHESIS_PROMPT_TEMPLATE = """\
ROLE
────
You are an expert AI Research Agent. You just performed a live web search and read several online sources. Write a clear, authoritative answer to the user's question using ONLY the facts gathered below.

STRICT WRITING RULES (NON-NEGOTIABLE)
─────────────────────────────────────
• NEVER use robotic meta-language like "based on the provided context", "the context mentions", "according to the text", or "there is no information in the context".
• Speak directly about the real world, the facts, and the events. (e.g. Instead of saying "The context does not mention X", say "There is no evidence or public report indicating X.")
• Use ONLY the facts provided in the Research Data below. Do NOT assume, extrapolate, or hallucinate outside facts.
• If the gathered data does not fully answer the user's question, clearly state what is currently known and what specific details remain unconfirmed or unreported in the search results.
• Structure your response cleanly with bullet points or short, readable paragraphs where appropriate.

QUESTION
────────
{rewritten_query}

RESEARCH DATA GATHERED
──────────────────────
{rag_context}

AUTHORITATIVE ANSWER
────────────────────\
"""


# ==============================================================================
# NODE 1 — Query Rewriter
# ==============================================================================

def query_rewriter_node(state: AgentState) -> dict:
    """
    Resolves ambiguous references in the user's query using the last N turns
    of conversation history. Outputs a fully self-contained query string.

    Example:
        History : "Who is Dharmendra Pradhan?" → "He is an Indian politician..."
        Input   : "Did he resign?"
        Output  : "Did Dharmendra Pradhan resign?"
    """
    original_query = state["original_query"]
    all_messages   = state["messages"]

    # Grab last N turns (each turn = 1 HumanMessage + 1 AIMessage = 2 messages)
    window_size     = REWRITER_CONTEXT_TURNS * 2
    recent_messages = all_messages[-(window_size + 1):-1]  # exclude current question

    # Format as readable conversation string
    recent_history = ""
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            recent_history += f"User : {msg.content}\n"
        elif isinstance(msg, AIMessage):
            recent_history += f"Agent: {msg.content}\n"

    # If no prior history, the query is already self-contained
    if not recent_history.strip():
        print(f"[Rewriter] No prior history — query unchanged: '{original_query}'")
        return {"rewritten_query": original_query}

    prompt = REWRITER_PROMPT_TEMPLATE.format(
        n=REWRITER_CONTEXT_TURNS,
        recent_history=recent_history.strip(),
        current_query=original_query,
    )

    response = web_search_agent.fast_model.invoke(prompt)
    rewritten = response.content.strip()

    print(f"[Rewriter] '{original_query}' → '{rewritten}'")
    return {"rewritten_query": rewritten}


# ==============================================================================
# NODE 2 — Search + Scrape + Index
# (No LLM — pure deterministic pipeline)
# ==============================================================================

def search_and_index_node(state: AgentState) -> dict:
    import time
    t_start = time.time()
    query = state["rewritten_query"]
    print(f"\n[Search & Index] Query: '{query}'")

    # ── 1. Fallback search chain ───────────────────────────────────────────────
    t0 = time.time()
    search_payload = web_search_agent.search_with_fallback(query)
    tool_used      = search_payload["search_engine_used"]
    print(f"  └─ Search API ({tool_used}) took: {round(time.time() - t0, 2)}s")

    if not search_payload["success"] or not search_payload["response"]:
        print("[Search & Index] ❌ All search tools failed.")
        return {"rag_results": "No information found — all search tools failed."}

    results = search_payload["response"]

    # ── 2. Index Tavily snippets immediately ─────────────────────────────────
    items_list = []
    if isinstance(results, dict) and "results" in results:
        items_list = results["results"]
    elif isinstance(results, list):
        items_list = results
    elif isinstance(results, str):
        rag_pipeline.add_text_documents(results, source=f"{tool_used} Search")

    urls_to_scrape = []
    for item in items_list:
        if not isinstance(item, dict):
            continue

        url     = item.get("url", "")
        content = item.get("content", "")

        # Safety net: Index snippets immediately
        if content:
            rag_pipeline.add_text_documents(content, source=f"Tavily Snippet | {url}")

        if url:
            urls_to_scrape.append(url)

    # ── 3. Parallel full-page scraping (Concurrently across 5 threads) ──────
    if urls_to_scrape:
        t_scrape = time.time()
        print(f"[Scraping] Launching parallel scrape for {len(urls_to_scrape)} URLs...")

        def _fetch(target_url):
            try:
                text = scrape_and_clean_url(target_url)
                return target_url, text
            except Exception as err:
                print(f"[Scraper] ⚠️ Failed {target_url}: {err}")
                return target_url, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(_fetch, url): url for url in urls_to_scrape}
            for future in as_completed(future_to_url):
                target_url, full_text = future.result()
                if full_text:
                    rag_pipeline.add_text_documents(full_text, source=f"Scraped Page | {target_url}")

        print(f"  └─ Parallel Scraping took: {round(time.time() - t_scrape, 2)}s")

    # ── 4. Retrieve top-k FAISS chunks ────────────────────────────────────────
    t_faiss = time.time()
    rag_context = rag_pipeline.retrieve(query, k=10)
    print(f"  └─ FAISS Retrieval took: {round(time.time() - t_faiss, 2)}s")
    print(f"[Search & Index] ✅ Retrieved {len(rag_context)} chars via {tool_used} (Total node: {round(time.time() - t_start, 2)}s)")

    return {"rag_results": rag_context}


# ==============================================================================
# NODE 3 — Synthesizer
# ==============================================================================

def synthesis_node(state: AgentState) -> dict:
    """
    Generates the final answer from a clean, isolated prompt.

    Only the rewritten query and retrieved context are passed to the LLM.
    The full message history (with its prior Q&A turns) is intentionally
    excluded to keep the synthesis prompt clean and focused.
    """
    rewritten_query = state["rewritten_query"]
    rag_context     = state["rag_results"]

    if not rag_context or rag_context.startswith("No information found"):
        final_answer = (
            "I wasn't able to find relevant information for your question. "
            "Please try rephrasing or ask about a different topic."
        )
    else:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            rewritten_query=rewritten_query,
            rag_context=rag_context,
        )
        response     = web_search_agent.model.invoke(prompt)
        final_answer = response.content.strip()

    # Add the final answer to message history so future turns can reference it
    return {
        "messages":     [AIMessage(content=final_answer)],
        "final_answer": final_answer,
    }