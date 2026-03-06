"""
JEE Algebra Solver — FastAPI Backend
=====================================
REST API for the multi-agent JEE maths tutoring system.

Endpoints:
  POST /solve              — primary JEE solve endpoint (text input)
  POST /solve/text         — alias for /solve
  POST /solve/image        — OCR → solve pipeline
  POST /solve/audio        — ASR → solve pipeline
  POST /extract/image      — OCR only (no solving)
  POST /feedback           — user feedback / correction
  GET  /history            — recent interactions
  POST /ingest             — ingest markdown/PDF knowledge base files
  POST /ingest/jee         — upsert curated JEE formula chunks
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agents.orchestrator import run_pipeline
from memory.memory_store import MemoryStore
from utils.config import FASTAPI_HOST, FASTAPI_PORT
from utils.logger import get_logger

logger = get_logger("app")

app = FastAPI(
    title="JEE Algebra Solver",
    description="Multi-agent AI tutoring system for JEE Advanced/Mains algebra",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────

class TextRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    interaction_id: int
    feedback: str          # 'correct' or 'incorrect'
    comment: str = ""
    corrected_answer: str = ""


class PipelineResponse(BaseModel):
    success: bool
    problem_text: str = ""
    topic: str = ""
    problem_type: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    score_breakdown: dict = {}
    explanation: dict = {}
    rag_sources: list = []
    solution_steps: list = []
    trace: list = []
    needs_human_review: bool = False
    hitl_type: str = ""
    interaction_id: int = 0
    status: str = "PARTIAL"
    retry_count: int = 0
    error: str = ""


class SolutionDetail(BaseModel):
    steps: list = []
    final_answer: str = ""
    answer_representation: str = ""


class VerificationDetail(BaseModel):
    verified: bool = False
    confidence_score: float = 0.0
    method: str = "hybrid"


class JEESolveResponse(BaseModel):
    problem: str = ""
    topic: str = ""
    problem_type: str = ""
    solution: SolutionDetail = Field(default_factory=SolutionDetail)
    verification: VerificationDetail = Field(default_factory=VerificationDetail)
    explanation: str = ""
    key_insight: str = ""
    status: str = "PARTIAL"
    rag_sources: list = []
    trace: list = []
    needs_human_review: bool = False
    interaction_id: int = 0
    retry_count: int = 0
    error: str = ""


@app.get("/")
def root():
    return {
        "name": "JEE Algebra Solver",
        "version": "2.0.0",
        "endpoints": [
            "POST /solve",
            "POST /solve/text",
            "POST /solve/image",
            "POST /solve/audio",
            "POST /extract/image",
            "POST /feedback",
            "GET  /history",
            "POST /ingest",
            "POST /ingest/jee",
        ],
    }


@app.post("/solve", response_model=JEESolveResponse)
def solve_jee(request: TextRequest):
    """
    Primary JEE solve endpoint.

    Accepts plain-text math problem; returns the canonical JEE output schema:
      problem, topic, problem_type, solution, verification, explanation,
      key_insight, status (SOLVED | UNSOLVED | PARTIAL | BLOCKED).
    """
    logger.info("JEE solve request: %s", request.text[:80])

    result = run_pipeline(
        input_type="text",
        raw_input=request.text,
        input_confidence=1.0,
    )

    expl = result.get("explanation", {})
    if isinstance(expl, str):
        expl = {"formatted_explanation": expl}
    
    conf = result.get("confidence", 0.0)

    return JEESolveResponse(
        problem=result.get("problem_text", request.text),
        topic=result.get("topic", ""),
        problem_type=result.get("problem_type", ""),
        solution=SolutionDetail(
            steps=result.get("solution_steps", []),
            final_answer=result.get("final_answer") or "Refer to the step-by-step Explanation below.",
            answer_representation=result.get("final_answer", ""),
        ),
        verification=VerificationDetail(
            verified=result.get("is_verified", conf >= 0.8),
            confidence_score=conf,
            method="hybrid",
        ),
        explanation=expl.get("formatted_explanation", ""),
        key_insight=expl.get("key_insight", ""),
        status=result.get("status", "PARTIAL"),
        rag_sources=result.get("rag_sources", []),
        trace=result.get("trace", []),
        needs_human_review=result.get("needs_human_review", False),
        interaction_id=result.get("interaction_id", 0),
        retry_count=result.get("retry_count", 0),
        error=result.get("error", ""),
    )


@app.post("/solve/text", response_model=PipelineResponse)
def solve_text(request: TextRequest):
    """Solve a math problem from text input (full detail response)."""
    logger.info("Received text request: %s", request.text[:80])

    result = run_pipeline(
        input_type="text",
        raw_input=request.text,
        input_confidence=1.0,
    )

    return _build_response(result)


@app.post("/solve/image", response_model=PipelineResponse)
async def solve_image(file: UploadFile = File(...)):
    """Solve a math problem from an uploaded image."""
    logger.info("Received image: %s", file.filename)

    # Save uploaded file to temp
    suffix = os.path.splitext(file.filename or ".png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # OCR extraction
        from utils.ocr_utils import extract_text_from_image

        ocr_result = extract_text_from_image(tmp_path)
        extracted_text = ocr_result["text"]
        ocr_confidence = ocr_result["confidence"]

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="OCR could not extract text from the image.")

        result = run_pipeline(
            input_type="image",
            raw_input=extracted_text,
            input_confidence=ocr_confidence,
        )

        response = _build_response(result)
        # Include OCR metadata
        response.trace.insert(0, {
            "step": 0,
            "agent_name": "OCR (EasyOCR)",
            "action": "Text extraction",
            "input_summary": file.filename,
            "output_summary": extracted_text[:100],
            "metadata": {"confidence": ocr_confidence},
        })
        # Output raw validation result
        return _build_response(result)

    finally:
        os.unlink(tmp_path)


@app.post("/extract/image")
async def extract_image(file: UploadFile = File(...)):
    """Extract text from an uploaded image without solving it."""
    logger.info("Received image for extraction: %s", file.filename)

    suffix = os.path.splitext(file.filename or ".png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from utils.ocr_utils import extract_text_from_image

        ocr_result = extract_text_from_image(tmp_path)
        extracted_text = ocr_result["text"]
        ocr_confidence = ocr_result["confidence"]

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="OCR could not extract text from the image.")

        return {
            "problem_text": extracted_text,
            "confidence": ocr_confidence
        }
    finally:
        os.unlink(tmp_path)


@app.post("/solve/audio", response_model=PipelineResponse)
async def solve_audio(file: UploadFile = File(...)):
    """Solve a math problem from an uploaded audio file."""
    logger.info("Received audio: %s", file.filename)

    suffix = os.path.splitext(file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from utils.audio_utils import transcribe_audio

        asr_result = transcribe_audio(tmp_path)
        transcript = asr_result["transcript"]
        asr_confidence = asr_result["confidence"]

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Whisper could not transcribe the audio.")

        result = run_pipeline(
            input_type="audio",
            raw_input=transcript,
            input_confidence=asr_confidence,
        )

        response = _build_response(result)
        response.trace.insert(0, {
            "step": 0,
            "agent_name": "ASR (Whisper)",
            "action": "Transcription",
            "input_summary": file.filename,
            "output_summary": transcript[:100],
            "metadata": {"confidence": asr_confidence},
        })
        return response

    finally:
        os.unlink(tmp_path)


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Submit user feedback / correction for an interaction."""
    logger.info(
        "Feedback for #%d: %s", request.interaction_id, request.feedback
    )

    store = MemoryStore()
    success = store.apply_correction(
        interaction_id=request.interaction_id,
        feedback=request.feedback,
        comment=request.comment,
        corrected_answer=request.corrected_answer,
    )

    if success:
        return {"status": "ok", "message": "Feedback saved. The system will learn from this correction."}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback.")


@app.get("/history")
def get_history(limit: int = 20):
    """Get recent interaction history."""
    store = MemoryStore()
    return {"history": store.get_history(limit=limit)}


@app.post("/ingest")
def ingest_knowledge():
    """Ingest markdown/PDF knowledge base files into the vector store."""
    from rag.ingest import ingest_knowledge_base
    result = ingest_knowledge_base()
    return {"status": "ok", "result": result}


@app.post("/ingest/jee")
def ingest_jee_knowledge():
    """Upsert the curated JEE formula chunks into the vector store."""
    from rag.ingest import build_jee_knowledge_base
    result = build_jee_knowledge_base()
    return {"status": "ok", "result": result}


# ── Helpers ───────────────────────────────────────

def _build_response(result: dict) -> PipelineResponse:
    """Convert pipeline state dict to legacy PipelineResponse."""
    return PipelineResponse(
        success=not bool(result.get("error")),
        problem_text=result.get("problem_text", ""),
        topic=result.get("topic", ""),
        problem_type=result.get("problem_type", ""),
        final_answer=result.get("final_answer", ""),
        confidence=result.get("confidence", 0.0),
        score_breakdown=result.get("score_breakdown", {}),
        explanation=result.get("explanation", {}),
        rag_sources=result.get("rag_sources", []),
        solution_steps=result.get("solution_steps", []),
        trace=result.get("trace", []),
        needs_human_review=result.get("needs_human_review", False),
        hitl_type=result.get("hitl_type", ""),
        interaction_id=result.get("interaction_id", 0),
        status=result.get("status", "PARTIAL"),
        retry_count=result.get("retry_count", 0),
        error=result.get("error", ""),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)
