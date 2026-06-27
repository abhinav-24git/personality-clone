import os
import sys

# Add root folder to sys.path for local module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Pipeline Imports
from src.inference.generator import CloneGenerator
from src.rag.ingest_docs import build_vector_store

app = FastAPI(
    title="Digital Clone Bot API",
    description="Backend for Abhinav's Digital Clone - Intent Routing, ChromaDB RAG, and Streaming Generation",
    version="1.0"
)

# Enable CORS for easy integration with frontend portfolios (e.g. Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global generator
# Can be overridden via env: CLONE_MODE=mock, hf_api, or local
CLONE_MODE = os.environ.get("CLONE_MODE", "mock")
generator = CloneGenerator(mode=CLONE_MODE)

class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    docs_dir: str = "data/rag_docs"
    db_dir: str = "data/vector_store"

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_mode": generator.mode,
        "chromadb_loaded": generator.collection is not None,
        "message": "Welcome to Abhinav's Digital Clone API. Send requests to /api/chat"
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Exposes an SSE (Server-Sent Events) streaming endpoint.
    Streams back words generated in real-time.
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    print(f"Received message: '{user_msg}'")
    
    def event_generator():
        try:
            # Yield tokens from the generator
            for token in generator.generate_response(user_msg):
                # SSE event format requires "data: <payload>\n\n"
                # We JSON-encode the payload to handle newlines and spaces safely
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"error": str(e)})
            yield f"data: {err_payload}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/ingest")
def trigger_ingestion(request: IngestRequest):
    """
    Manually triggers RAG ingestion. 
    Reads documents from the specified directory and rebuilds the ChromaDB store.
    """
    try:
        collection = build_vector_store(request.docs_dir, request.db_dir)
        if collection is not None:
            # Refresh generator's vector store connection
            global generator
            generator = CloneGenerator(mode=generator.mode)
            return {"status": "success", "message": f"Successfully indexed documents into ChromaDB."}
        else:
            return {"status": "error", "message": "Failed to build vector store. Check logs for details."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run server locally on port 8001
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting Digital Clone API in {CLONE_MODE.upper()} mode on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
