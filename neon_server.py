"""
NEON Agent HTTP Server
======================

REST API for the Neon Health hiring challenge agent.

Run with: python neon_server.py
Or with: uvicorn neon_server:app --reload

Endpoints:
  POST /challenge  - Process a challenge message
  POST /reset      - Reset agent memory
  GET  /health     - Health check
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Use OpenAI agent if API key is available, otherwise fall back to rule-based
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = OPENAI_KEY is not None and len(OPENAI_KEY) > 20

if USE_OPENAI:
    try:
        from neon_agent_openai import NeonAgent, reconstruct_message
        print(f"Using OpenAI-powered agent (key: {OPENAI_KEY[:8]}...)")
    except Exception as e:
        print(f"Failed to load OpenAI agent: {e}, falling back to rule-based")
        from neon_agent import NeonAgent, reconstruct_message
        USE_OPENAI = False
else:
    from neon_agent import NeonAgent, reconstruct_message
    print(f"Using rule-based agent (OPENAI_API_KEY not set or invalid, got: {OPENAI_KEY[:8] if OPENAI_KEY else 'None'}...)")

# =============================================================================
# API MODELS
# =============================================================================

class ChallengeRequest(BaseModel):
    type: str  # Should be "challenge"
    message: List[Dict[str, Any]]  # Fragments: [{"word": str, "timestamp": num}, ...]


class NeonCodeRequest(BaseModel):
    code: str


class AgentResponse(BaseModel):
    type: str  # "enter_digits" or "speak_text"
    text: Optional[str] = None
    digits: Optional[str] = None


# =============================================================================
# LOAD RESUME
# =============================================================================

def load_default_resume():
    """Load the default resume from nithin_resume.json"""
    resume_path = os.path.join(os.path.dirname(__file__), "nithin_resume.json")
    print(f"Looking for resume at: {resume_path}")

    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            data = json.load(f)
            print(f"Resume loaded successfully: {data.get('name', 'Unknown')}")
            return data

    print("Resume file not found, using fallback")
    # Fallback resume
    return {
        "name": "Nithin R",
        "summary": "Software Development Engineer at Amazon specializing in Gen AI, ML systems, and billion-scale vector search.",
        "education": [{"degree": "Masters", "field": "Computer Information Systems", "school": "Central Michigan University", "year": "2024"}],
        "experience": [{"title": "Software Development Engineer", "company": "Amazon.com, Inc.", "description": "Built billion-scale vector similarity search platforms using Faiss-HNSW. Invented patent-pending online vector search architecture."}],
        "projects": [{"name": "RAG Agents", "description": "Built RAG chatbot with GPT-4", "technologies": ["Python", "Qdrant", "GPT-4"]}],
        "skills": ["Python", "PyTorch", "FAISS", "AWS", "RAG", "LLMs", "Distributed Systems", "Gen AI"]
    }


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="NEON Agent API",
    description="Stateful AI agent for the Neon Health hiring challenge",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = NeonAgent()

# Load default resume
resume_data = load_default_resume()
if resume_data:
    agent.load_resume_dict(resume_data)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/challenge", response_model=AgentResponse)
async def process_challenge(request: ChallengeRequest):
    """
    Process a challenge message.

    Input:  {"type": "challenge", "message": [{"word": str, "timestamp": num}, ...]}
    Output: {"type": "enter_digits", "digits": "..."} or {"type": "speak_text", "text": "..."}
    """
    try:
        challenge = {"type": request.type, "message": request.message}
        response = agent.process(challenge)
        return AgentResponse(**response)
    except Exception as e:
        return AgentResponse(type="speak_text", text=f"Error: {str(e)}")


@app.post("/process")
async def process_prompt(request: dict):
    """
    Legacy endpoint - accepts prompt string or fragments.
    """
    try:
        # If it's a challenge format
        if "type" in request and request["type"] == "challenge":
            response = agent.process(request)
            return response

        # If it's fragments
        if "fragments" in request:
            challenge = {"type": "challenge", "message": request["fragments"]}
            response = agent.process(challenge)
            return response

        # If it's a simple prompt
        if "prompt" in request:
            response = agent.process_raw(request["prompt"])
            return response

        return {"type": "speak_text", "text": "Invalid request format"}
    except Exception as e:
        return {"type": "speak_text", "text": f"Error: {str(e)}"}


@app.post("/set-code")
async def set_neon_code(request: NeonCodeRequest):
    """Set the Neon authorization code."""
    code = request.code.strip()

    # Validate code: must be 16 alphanumeric characters
    if not code:
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    if len(code) != 16:
        raise HTTPException(status_code=400, detail="Code must be exactly 16 characters")
    if not code.isalnum():
        raise HTTPException(status_code=400, detail="Code must be alphanumeric")

    agent.set_neon_code(code)
    return {"status": "success", "message": "Neon code set"}


@app.post("/reset")
async def reset_agent():
    """Reset agent memory."""
    agent.reset()
    return {"status": "success", "message": "Agent memory cleared"}


@app.post("/reset-full")
async def reset_full():
    """Full reset: clear memory AND code."""
    global agent
    agent = NeonAgent()
    resume_data = load_default_resume()
    if resume_data:
        agent.load_resume_dict(resume_data)
    return {"status": "success", "message": "Agent fully reset"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "neon_code_set": agent.neon_code is not None,
        "resume_loaded": agent.resume is not None,
        "crew_manifests_stored": len(agent.memory.transmissions)
    }


@app.get("/memory")
async def get_memory():
    """Get stored crew manifest transmissions."""
    return {
        "crew_manifests": agent.memory.transmissions,
        "all_responses": agent.memory.all_responses
    }


# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "NEON Agent API v2.0. Visit /docs for API documentation."}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("NEON AGENT SERVER v2.0")
    print("=" * 50)
    print("Frontend UI:  http://localhost:8000")
    print("API Docs:     http://localhost:8000/docs")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
