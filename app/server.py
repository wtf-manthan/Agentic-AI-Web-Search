import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Trigger one-time warmup of embedding models and search pipeline into RAM
print("[Server] Initializing Web Search Agent pipeline and vector store...")
from workflow.graph import app as graph_app
print("[Server] ✅ Search engine initialized and ready in system memory.")

app = FastAPI(title="Web Search Agent API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None

class ChatResponse(BaseModel):
    original_query: str
    rewritten_query: str
    rag_results: str
    final_answer: str
    thread_id: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "messages": [HumanMessage(content=req.question)],
        "original_query": req.question,
        "rewritten_query": "",
        "rag_results": "",
        "final_answer": "",
    }
    
    import time
    start_t = time.time()
    print(f"\n[Search API] Executing query for session: {thread_id[:8]}...")
    try:
        result = graph_app.invoke(state, config=config)
        elapsed = round(time.time() - start_t, 2)
        print(f"[Search API] ⏱️ Total query time: {elapsed}s")
        return ChatResponse(
            original_query=req.question,
            rewritten_query=result.get("rewritten_query", req.question),
            rag_results=result.get("rag_results", ""),
            final_answer=result.get("final_answer", ""),
            thread_id=thread_id
        )
    except Exception as e:
        print(f"[Search API] ❌ Execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "online", "engine": "ready"}

# Serve the static UI files from /
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Web Search Agent - Server Online")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
