"""
HTTP API Server for Neon Agent
==============================

This provides a REST API interface to the agent.

Run with: python server.py
Or with: uvicorn server:app --reload

Endpoints:
  POST /process - Process a prompt and get a response
  POST /reset   - Reset the agent's memory
  GET  /memory  - Get all stored responses
  POST /load-resume - Load a resume
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json

from agent import NeonAgent, create_agent_with_resume

# =============================================================================
# API MODELS
# =============================================================================

class PromptRequest(BaseModel):
    prompt: str


class ResumeRequest(BaseModel):
    resume: Dict[str, Any]


class FragmentedRequest(BaseModel):
    fragments: List[Dict[str, Any]]


class AgentResponse(BaseModel):
    type: str
    text: Optional[str] = None
    digits: Optional[str] = None


class MemoryEntry(BaseModel):
    index: int
    task_type: str
    prompt: str
    response_text: str
    response_json: Dict[str, str]


# =============================================================================
# LOAD RESUME FROM JSON FILE
# =============================================================================

import os

def load_default_resume():
    """Load the default resume from nithin_resume.json"""
    resume_path = os.path.join(os.path.dirname(__file__), "nithin_resume.json")
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            return json.load(f)
    # Fallback if file not found
    return {
        "name": "Nithin R",
        "summary": "Software Development Engineer at Amazon specializing in Gen AI, ML systems, and billion-scale vector search.",
        "education": [{"degree": "Masters", "field": "Computer Information Systems", "school": "Central Michigan University", "year": "2024"}],
        "experience": [{"title": "Software Development Engineer", "company": "Amazon.com, Inc.", "description": "Built billion-scale vector search platforms"}],
        "projects": [{"name": "RAG Agents", "description": "Built RAG chatbot with GPT-4", "technologies": ["Python", "Qdrant", "GPT-4"]}],
        "skills": ["Python", "PyTorch", "FAISS", "AWS", "RAG", "LLMs", "Distributed Systems"]
    }

DEFAULT_RESUME = load_default_resume()

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Neon Agent API",
    description="Stateful AI agent for the Neon Health hiring challenge",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global agent instance (stateful across requests)
agent = create_agent_with_resume(DEFAULT_RESUME)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/process", response_model=AgentResponse)
async def process_prompt(request: PromptRequest):
    """
    Process a prompt and return a response.

    The response will always be one of:
    - {"type": "speak_text", "text": "..."}
    - {"type": "enter_digits", "digits": "..."}
    """
    try:
        response = agent.process(request.prompt)
        return AgentResponse(**response)
    except Exception as e:
        # Even errors must return valid format
        return AgentResponse(type="speak_text", text=f"Error: {str(e)}")


@app.post("/process-fragments", response_model=AgentResponse)
async def process_fragments(request: FragmentedRequest):
    """
    Process fragmented input (will be sorted by timestamp and joined).

    Input format:
    {
        "fragments": [
            {"timestamp": 1.0, "word": "Hello"},
            {"timestamp": 2.0, "word": "World"}
        ]
    }
    """
    try:
        # Convert fragments to JSON string for agent processing
        prompt = json.dumps(request.fragments)
        response = agent.process(prompt)
        return AgentResponse(**response)
    except Exception as e:
        return AgentResponse(type="speak_text", text=f"Error: {str(e)}")


@app.post("/load-resume")
async def load_resume(request: ResumeRequest):
    """Load a new resume into the agent."""
    try:
        agent.load_resume_from_dict(request.resume)
        return {"status": "success", "message": "Resume loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset")
async def reset_memory():
    """Reset the agent's memory (clear all stored responses)."""
    agent.reset_memory()
    return {"status": "success", "message": "Memory cleared"}


@app.post("/reset-full")
async def reset_full():
    """Full reset: clear memory AND reset authentication state."""
    global agent
    agent = create_agent_with_resume(DEFAULT_RESUME)
    return {"status": "success", "message": "Agent fully reset (memory + auth)"}


@app.get("/memory", response_model=List[MemoryEntry])
async def get_memory():
    """Get all stored responses from memory."""
    return [MemoryEntry(**entry) for entry in agent.get_memory_dump()]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "authenticated": agent.is_authenticated(),
        "memory_size": len(agent.memory.entries),
        "resume_loaded": agent.resume is not None
    }


class NeonCodeRequest(BaseModel):
    code: str


@app.post("/authenticate")
async def authenticate(request: NeonCodeRequest):
    """Set the Neon authorization code."""
    success = agent.set_neon_code(request.code)
    if success:
        return {"status": "success", "message": "Authentication successful"}
    else:
        raise HTTPException(status_code=400, detail="Invalid Neon code")


@app.get("/auth-status")
async def auth_status():
    """Check authentication status."""
    return {
        "authenticated": agent.is_authenticated(),
        "neon_code_set": agent.neon_code is not None
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Neon Agent API. Visit /docs for API documentation."}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("NEON AGENT SERVER")
    print("=" * 50)
    print("Frontend UI:  http://localhost:8000")
    print("API Docs:     http://localhost:8000/docs")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
