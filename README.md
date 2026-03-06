# JEE Algebra Solver — Multi-Agent System

A production-grade multi-agent system that solves JEE (Joint Entrance Examination) algebra problems end-to-end. Every computation is deterministic (SymPy); the LLM is used only to parse the problem and format the explanation.

---

## Architecture

```
User Input (text / image / audio)
        │
        ▼
┌──────────────────────┐
│   Parser Agent        │  Extracts canonical problem text from any modality
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Router Agent        │  Classifies problem type (10 JEE algebra topics)
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────┐
│ RAG     │  │ Solver   │  ReAct loop → dispatches SymPy tools
│Retriever│  │  Agent   │
└─────────┘  └────┬─────┘
                  │
                  ▼
          ┌──────────────┐
          │  Verifier    │  5 checks: substitution, Vieta's, discriminant,
          │   Agent      │  boundary, numeric cross-check (confidence score)
          └──────┬───────┘
                 │ (retry loop up to 3×)
                 ▼
          ┌──────────────┐
          │  Explainer   │  JEE-format step-by-step explanation (LLM)
          │   Agent      │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  Memory      │  Persists Q&A history (in-memory + extractable)
          │   Store      │
          └──────────────┘
```

---

## Directory Structure

```
Maths_agent/
│
├── app.py                     # FastAPI backend — all HTTP endpoints
├── demo.py                    # Quick smoke-test / demo runner
├── generate_samples.py        # Generates sample JEE problems for testing
├── requirements.txt           # Python dependencies
│
├── agents/                    # All 7 multi-agent pipeline stages
│   ├── __init__.py
│   ├── orchestrator.py        # LangGraph-style pipeline + retry loop (up to 3×)
│   ├── parser_agent.py        # Multimodal input → canonical problem text
│   ├── router_agent.py        # Problem-type classifier (10 JEE algebra topics)
│   ├── solver_agent.py        # ReAct loop — calls SymPy tools via dispatch_tool
│   ├── verifier_agent.py      # 5 deterministic checks + hybrid confidence score
│   ├── explainer_agent.py     # JEE-format student explanation generator
│   └── guardrail_agent.py     # Input/output safety guardrails
│
├── tools/
│   ├── __init__.py
│   ├── math_tools.py          # Math Tool Executor — 20 SymPy tools + dispatch_tool
│   ├── calculator.py          # Lightweight expression evaluator (legacy)
│   └── sympy_solver.py        # Direct SymPy solver utilities (legacy)
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py              # Knowledge base ingestion (file-based + 25 JEE chunks)
│   └── retriever.py           # ChromaDB semantic retriever
│
├── memory/
│   ├── __init__.py
│   └── memory_store.py        # In-process conversation memory
│
├── utils/
│   ├── __init__.py
│   ├── config.py              # Centralised config (env vars, thresholds)
│   ├── logger.py              # Structured logger
│   ├── trace_logger.py        # Per-request trace/audit logger
│   ├── json_parser.py         # Robust JSON extraction from LLM responses
│   ├── audio_utils.py         # Whisper-based audio transcription
│   └── ocr_utils.py           # Image → text (Tesseract / Gemini Vision)
│
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py       # Streamlit frontend (confidence meter, JEE tabs)
│
├── data/
│   ├── chroma_db/             # Persistent ChromaDB vector store
│   └── knowledge_base/        # Markdown reference documents
│       ├── algebra.md
│       ├── calculus.md
│       ├── common_mistakes.md
│       ├── jee_formulas.md    # Comprehensive JEE formula reference (250+ lines)
│       ├── linear_algebra.md
│       ├── probability.md
│       └── problem_solving_templates.md
│
├── tests/
│   └── test_golden.py         # 8 golden JEE cases + 7 SymPy sanity checks
│
└── logs/                      # Runtime logs (auto-created)
```

---

## Supported Problem Types

| # | Type | Example |
|---|------|---------|
| 1 | Quadratic equations | x² − 5x + 6 = 0 |
| 2 | Polynomial equations | x³ − 6x² + 11x − 6 = 0 |
| 3 | Inequalities | x² − 4x + 3 < 0 |
| 4 | Arithmetic Progressions | Sum of 2, 5, 8, … (n=10) |
| 5 | Geometric Progressions | Sum of 1, 2, 4, … (n=5) |
| 6 | Permutation & Combination | C(10, 3) |
| 7 | Binomial Theorem | Coefficient of x² in (x+1)⁴ |
| 8 | Matrices & Determinants | det([[2,3],[1,4]]) |
| 9 | Probability | P(2 red from 2R+3B without replacement) |
| 10 | Complex Numbers | Modulus, argument, De Moivre |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/solve` | Full JEE pipeline → `JEESolveResponse` |
| `POST` | `/solve/text` | Legacy text endpoint → `PipelineResponse` |
| `POST` | `/solve/image` | Image input via multipart upload |
| `POST` | `/solve/audio` | Audio input via multipart upload |
| `POST` | `/ingest` | Ingest files from `data/knowledge_base/` |
| `POST` | `/ingest/jee` | Ingest built-in 25 JEE formula chunks |
| `GET`  | `/health` | Health check |
| `GET`  | `/history` | Session conversation history |

### `POST /solve` — Response Schema

```json
{
  "problem": "x² - 5x + 6 = 0",
  "topic": "quadratic_equations",
  "problem_type": "quadratic_standard_form",
  "solution": {
    "steps": ["Step 1: ...", "Step 2: ..."],
    "final_answer": "x = 2, x = 3",
    "answer_representation": "x ∈ {2, 3}"
  },
  "verification": {
    "verified": true,
    "confidence_score": 0.95,
    "method": "substitution + vieta"
  },
  "explanation": "📌 **Concept** ...\n📚 **Formula** ...\n🔢 **Steps** ...",
  "key_insight": "Sum of roots = 5, product = 6",
  "status": "SOLVED",
  "retry_count": 0,
  "rag_sources": [],
  "processing_time_ms": 1234
}
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gemini-2.5-pro
CONFIDENCE_THRESHOLD=0.8
MAX_RETRIES=3
CHROMA_DB_PATH=data/chroma_db
KNOWLEDGE_BASE_PATH=data/knowledge_base
```

### 3. Ingest the knowledge base

```bash
curl -X POST http://localhost:8000/ingest/jee
```

### 4. Run the backend

```bash
uvicorn app:app --reload --port 8000
```

### 5. Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

---

## Running Tests

```bash
pytest tests/test_golden.py -v
```

Expected: **15 passed** in < 1 s.

---

## Math Tool Executor

All 20 tools in `tools/math_tools.py` are callable via `dispatch_tool`:

```python
from tools.math_tools import dispatch_tool

result = dispatch_tool("solve_equation", expression="x**2 - 5*x + 6", variable="x")
# → {"success": True, "solutions": ["2", "3"], "solutions_numeric": [2.0, 3.0]}
```

| Tool | Key Parameters | Primary Return Keys |
|------|---------------|---------------------|
| `solve_equation` | `expression`, `variable` | `solutions`, `solutions_numeric` |
| `solve_inequality` | `expression`, `variable` | `solution_set` |
| `find_range` | `expression`, `variable`, `domain_str` | `range` |
| `factor_polynomial` | `expression` | `factored` |
| `simplify_expression` | `expression` | `simplified` |
| `expand_expression` | `expression` | `expanded` |
| `compute_determinant` | `matrix: list[list]` | `determinant_numeric` |
| `compute_matrix_inverse` | `matrix: list[list]` | `inverse` |
| `compute_eigenvalues` | `matrix: list[list]` | `eigenvalues` |
| `solve_linear_system` | `equations`, `variables` | `solution` |
| `compute_probability` | `favorable`, `total` | `probability_decimal` |
| `compute_combination` | `n`, `r` | `result` |
| `compute_permutation` | `n`, `r` | `result` |
| `compute_binomial_term` | `n`, `r`, `expression` | `coefficient` |
| `compute_ap_sum` | `first_term`, `common_diff`, `n` | `sum`, `nth_term` |
| `compute_gp_sum` | `first_term`, `common_ratio`, `n` | `sum` |
| `compute_hcf_lcm` | `numbers: list[int]` | `hcf`, `lcm` |
| `verify_solution` | `expression`, `variable`, `value` | `is_valid` |
| `verify_vieta` | `roots`, `coeff_a`, `coeff_b`, `coeff_c` | `all_pass`, `checks` |
| `evaluate_expression` | `expression`, `substitutions` | `value` |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `gemini-2.5-pro` | LLM model identifier |
| `OPENAI_API_KEY` | — | OpenAI-compatible API key |
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum verifier score before retry |
| `MAX_RETRIES` | `3` | Pipeline retry limit |
| `CHROMA_DB_PATH` | `data/chroma_db` | ChromaDB persistent storage path |
| `KNOWLEDGE_BASE_PATH` | `data/knowledge_base` | Markdown documents root |

---

## License

MIT License — free for educational and commercial use.
