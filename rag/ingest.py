"""
RAG Ingestion Pipeline — JEE Algebra Solver
=============================================
Two ingestion modes:
  1. `ingest_knowledge_base()` — reads all *.md / *.pdf files from the
     knowledge_base directory and chunks them into ChromaDB.
  2. `build_jee_knowledge_base()` — directly upserts a hand-crafted set of
     JEE-specific named chunks (formulas, theorems, templates) with rich
     topic/tag metadata for precision retrieval.

Run directly:
  python -m rag.ingest          (runs both modes)
"""

import os
import glob
from utils.config import CHROMA_PERSIST_DIR, KNOWLEDGE_BASE_DIR
from utils.logger import get_logger

logger = get_logger("rag.ingest")


def _get_embedding_function():
    """
    Return ChromaDB's built-in default embedding function (all-MiniLM-L6-v2
    via onnxruntime) — no full sentence-transformers install needed.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _get_or_create_collection(client, name: str = "math_knowledge"):
    """Get or create the ChromaDB collection with a consistent embedding function."""
    ef = _get_embedding_function()
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"description": "JEE math formulas, theorems, and problem-solving templates"},
    )


# ──────────────────────────────────────────────
# JEE structured knowledge chunks
# ──────────────────────────────────────────────
_JEE_CHUNKS: list[dict] = [
    # ---------- Quadratic Equations ----------
    {
        "id": "quad_formula",
        "topic": "quadratic_equations",
        "tags": ["quadratic", "formula", "roots", "discriminant"],
        "content": (
            "Quadratic Formula: For ax² + bx + c = 0, the roots are "
            "x = (-b ± √(b²-4ac)) / (2a). "
            "Discriminant D = b²-4ac: D>0 → two distinct real roots, "
            "D=0 → one repeated root x=-b/(2a), D<0 → complex conjugate roots."
        ),
    },
    {
        "id": "vieta_quadratic",
        "topic": "quadratic_equations",
        "tags": ["vieta", "quadratic", "sum_of_roots", "product_of_roots"],
        "content": (
            "Vieta's Formulas for ax²+bx+c=0 with roots α,β: "
            "α+β = -b/a (sum of roots), αβ = c/a (product of roots). "
            "Derived: α²+β² = (α+β)²-2αβ = b²/a²-2c/a. "
            "|α-β| = √D/|a| where D = b²-4ac."
        ),
    },
    {
        "id": "vieta_cubic",
        "topic": "polynomial_equations",
        "tags": ["vieta", "cubic", "sum_of_roots"],
        "content": (
            "Vieta's Formulas for cubic ax³+bx²+cx+d=0 with roots α,β,γ: "
            "α+β+γ = -b/a, αβ+βγ+γα = c/a, αβγ = -d/a."
        ),
    },
    # ---------- Inequalities ----------
    {
        "id": "am_gm",
        "topic": "inequalities",
        "tags": ["AM-GM", "inequality", "non-negative"],
        "content": (
            "AM-GM Inequality: For non-negative reals a₁,...,aₙ: "
            "(a₁+...+aₙ)/n ≥ (a₁·...·aₙ)^(1/n). "
            "Equality holds iff all aᵢ are equal."
        ),
    },
    {
        "id": "cauchy_schwarz",
        "topic": "inequalities",
        "tags": ["Cauchy-Schwarz", "inequality"],
        "content": (
            "Cauchy-Schwarz Inequality: "
            "(Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²). "
            "Equality iff aᵢ/bᵢ is constant for all i."
        ),
    },
    {
        "id": "quadratic_inequality",
        "topic": "inequalities",
        "tags": ["inequality", "quadratic", "sign_analysis"],
        "content": (
            "Quadratic Inequality (a>0): "
            "ax²+bx+c>0 is true for x<α or x>β when D>0 (α≤β are roots). "
            "ax²+bx+c>0 for ALL real x when D<0. "
            "ax²+bx+c<0 for α<x<β when a>0, D>0."
        ),
    },
    # ---------- Sequences & Series ----------
    {
        "id": "ap_formulas",
        "topic": "progressions",
        "tags": ["AP", "arithmetic_progression", "sum"],
        "content": (
            "Arithmetic Progression: aₙ = a+(n-1)d. "
            "Sₙ (sum of n terms) = n/2·[2a+(n-1)d] = n/2·(a+l) where l=last term. "
            "d = common difference = aₙ-aₙ₋₁."
        ),
    },
    {
        "id": "gp_formulas",
        "topic": "progressions",
        "tags": ["GP", "geometric_progression", "sum", "infinite"],
        "content": (
            "Geometric Progression: aₙ = ar^(n-1). "
            "Sₙ = a(rⁿ-1)/(r-1) for r≠1; Sₙ=na for r=1. "
            "S∞ = a/(1-r) for |r|<1 (infinite GP). "
            "r = common ratio = aₙ/aₙ₋₁."
        ),
    },
    {
        "id": "am_gm_hm",
        "topic": "progressions",
        "tags": ["AM", "GM", "HM", "inequality"],
        "content": (
            "For positive reals a, b: "
            "AM = (a+b)/2, GM = √(ab), HM = 2ab/(a+b). "
            "AM ≥ GM ≥ HM, equality iff a=b. "
            "AM·HM = GM²."
        ),
    },
    {
        "id": "sum_formulas",
        "topic": "progressions",
        "tags": ["sum", "series", "natural_numbers"],
        "content": (
            "Standard sums: Σk = n(n+1)/2, "
            "Σk² = n(n+1)(2n+1)/6, "
            "Σk³ = [n(n+1)/2]². "
            "All sums from k=1 to n."
        ),
    },
    # ---------- Permutations & Combinations ----------
    {
        "id": "ncr_npr",
        "topic": "permutation_combination",
        "tags": ["nCr", "nPr", "combinations", "permutations"],
        "content": (
            "Permutations: nPr = n!/(n-r)!. "
            "Combinations: nCr = n!/(r!(n-r)!). "
            "Pascal's rule: nCr + nC(r-1) = (n+1)Cr. "
            "Symmetry: nCr = nC(n-r)."
        ),
    },
    {
        "id": "derangement",
        "topic": "permutation_combination",
        "tags": ["derangement", "permutation", "no_fixed_point"],
        "content": (
            "Derangement D(n) = n! · Σ(-1)^k/k! for k=0..n. "
            "D(1)=0, D(2)=1, D(3)=2, D(4)=9, D(5)=44. "
            "Probability of a random permutation being a derangement → 1/e ≈ 0.368."
        ),
    },
    {
        "id": "circular_permutation",
        "topic": "permutation_combination",
        "tags": ["circular", "permutation"],
        "content": (
            "Circular permutation of n distinct objects = (n-1)!. "
            "If clockwise = anticlockwise = (n-1)!/2."
        ),
    },
    # ---------- Binomial Theorem ----------
    {
        "id": "binomial_expansion",
        "topic": "binomial_theorem",
        "tags": ["binomial", "expansion", "general_term"],
        "content": (
            "Binomial Theorem: (x+y)ⁿ = Σ nCr·x^(n-r)·y^r for r=0..n. "
            "General term: T(r+1) = nCr·x^(n-r)·y^r. "
            "Middle term: T(n/2+1) if n even; T((n+1)/2) and T((n+3)/2) if n odd. "
            "Sum of coefficients = 2ⁿ (set x=y=1)."
        ),
    },
    {
        "id": "binomial_coeff",
        "topic": "binomial_theorem",
        "tags": ["binomial", "coefficient", "sum"],
        "content": (
            "Binomial coefficient identities: Σ nCr = 2ⁿ, "
            "Σ nCr (r even) = Σ nCr (r odd) = 2^(n-1). "
            "(1+x)ⁿ ≈ 1+nx for |x|<<1."
        ),
    },
    # ---------- Matrices & Determinants ----------
    {
        "id": "determinant_2x2",
        "topic": "matrices_determinants",
        "tags": ["determinant", "2x2", "matrix"],
        "content": (
            "2×2 determinant: det([[a,b],[c,d]]) = ad-bc. "
            "Inverse: A⁻¹ = (1/det(A))·[[d,-b],[-c,a]] when det(A)≠0."
        ),
    },
    {
        "id": "determinant_3x3",
        "topic": "matrices_determinants",
        "tags": ["determinant", "3x3", "matrix", "cofactor"],
        "content": (
            "3×3 determinant by cofactor expansion along row 1: "
            "det([[a,b,c],[d,e,f],[g,h,i]]) = a(ei-fh) - b(di-fg) + c(dh-eg). "
            "det(AB)=det(A)·det(B). det(Aᵀ)=det(A). "
            "Swapping two rows negates the determinant."
        ),
    },
    {
        "id": "cramer_rule",
        "topic": "matrices_determinants",
        "tags": ["cramer", "linear_system", "determinant"],
        "content": (
            "Cramer's Rule: For Ax=b, xᵢ = det(Aᵢ)/det(A) where Aᵢ is A "
            "with ith column replaced by b. "
            "System is consistent iff rank(A)=rank([A|b])."
        ),
    },
    {
        "id": "cayley_hamilton",
        "topic": "matrices_determinants",
        "tags": ["Cayley-Hamilton", "characteristic_equation", "matrix"],
        "content": (
            "Cayley-Hamilton: Every matrix satisfies its own characteristic equation. "
            "For 2×2 matrix A: A² - tr(A)·A + det(A)·I = 0. "
            "Eigenvalues λ satisfy det(A-λI)=0."
        ),
    },
    # ---------- Probability ----------
    {
        "id": "probability_basics",
        "topic": "probability",
        "tags": ["probability", "axioms", "addition_rule"],
        "content": (
            "Probability axioms: 0≤P(A)≤1, P(S)=1. "
            "P(A∪B) = P(A)+P(B)-P(A∩B). "
            "P(Aᶜ) = 1-P(A). "
            "Mutually exclusive: P(A∪B)=P(A)+P(B)."
        ),
    },
    {
        "id": "conditional_bayes",
        "topic": "probability",
        "tags": ["conditional_probability", "Bayes", "independence"],
        "content": (
            "Conditional probability: P(A|B) = P(A∩B)/P(B). "
            "Independent events: P(A∩B) = P(A)·P(B). "
            "Bayes' theorem: P(Aᵢ|B) = P(B|Aᵢ)·P(Aᵢ) / ΣP(B|Aⱼ)·P(Aⱼ)."
        ),
    },
    {
        "id": "hypergeometric",
        "topic": "probability",
        "tags": ["hypergeometric", "without_replacement", "sampling"],
        "content": (
            "Hypergeometric probability (without replacement): "
            "P(X=k) = C(K,k)·C(N-K,n-k) / C(N,n). "
            "N=population, K=successes in population, n=sample size, k=desired successes."
        ),
    },
    {
        "id": "binomial_distribution",
        "topic": "probability",
        "tags": ["binomial_distribution", "with_replacement", "Bernoulli"],
        "content": (
            "Binomial distribution (with replacement, independent trials): "
            "P(X=k) = nCk·p^k·(1-p)^(n-k). "
            "Mean=np, Variance=np(1-p). "
            "n=trials, p=success probability, k=successes."
        ),
    },
    # ---------- Complex Numbers ----------
    {
        "id": "complex_basics",
        "topic": "complex_numbers",
        "tags": ["complex", "modulus", "argument", "conjugate"],
        "content": (
            "Complex number z=a+bi: |z|=√(a²+b²), arg(z)=arctan(b/a). "
            "Conjugate z̄=a-bi, |z|²=z·z̄. "
            "Re(z)=(z+z̄)/2, Im(z)=(z-z̄)/(2i)."
        ),
    },
    {
        "id": "de_moivre",
        "topic": "complex_numbers",
        "tags": ["De_Moivre", "complex", "roots_of_unity"],
        "content": (
            "De Moivre: (cosθ+isinθ)ⁿ = cos(nθ)+isin(nθ). "
            "nth roots of unity: e^(2πik/n) for k=0..n-1. "
            "Sum of nth roots of unity = 0 (n≥2). "
            "Cube roots: ω=e^(2πi/3), 1+ω+ω²=0, ω³=1."
        ),
    },
    # ---------- Number System ----------
    {
        "id": "hcf_lcm",
        "topic": "number_system",
        "tags": ["HCF", "LCM", "divisibility"],
        "content": (
            "HCF(a,b)×LCM(a,b) = a×b. "
            "Logarithm laws: log(xy)=log(x)+log(y), log(x/y)=log(x)-log(y), "
            "log(xᵃ)=a·log(x), change of base: log_b(x)=log_a(x)/log_a(b). "
            "log_b(1)=0, log_b(b)=1."
        ),
    },
]


def build_jee_knowledge_base() -> dict:
    """
    Directly upsert the curated JEE formula chunks into ChromaDB.
    Each chunk has rich metadata: topic, tags (comma-separated string).

    Returns:
        dict with chunk count and collection name.
    """
    import chromadb

    abs_persist = os.path.abspath(CHROMA_PERSIST_DIR)
    os.makedirs(abs_persist, exist_ok=True)
    client = chromadb.PersistentClient(path=abs_persist)
    collection = _get_or_create_collection(client)

    ids = []
    documents = []
    metadatas = []

    for chunk in _JEE_CHUNKS:
        ids.append(chunk["id"])
        documents.append(chunk["content"])
        metadatas.append(
            {
                "source": "jee_knowledge_base",
                "topic": chunk["topic"],
                "tags": ",".join(chunk["tags"]),
                "chunk_index": 0,
            }
        )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Upserted %d JEE knowledge chunks into 'math_knowledge'.", len(ids))
    return {"chunks": len(ids), "collection": "math_knowledge"}


def ingest_knowledge_base() -> dict:
    """
    Ingest all markdown/PDF files from the knowledge base directory into
    ChromaDB (file-based chunking mode, complements build_jee_knowledge_base).

    Returns:
        dict: files processed, chunks ingested, collection name.
    """
    import chromadb

    abs_persist = os.path.abspath(CHROMA_PERSIST_DIR)
    os.makedirs(abs_persist, exist_ok=True)
    client = chromadb.PersistentClient(path=abs_persist)
    collection = _get_or_create_collection(client)

    abs_kb_dir = os.path.abspath(KNOWLEDGE_BASE_DIR)
    md_files = glob.glob(os.path.join(abs_kb_dir, "*.md"))
    pdf_files = glob.glob(os.path.join(abs_kb_dir, "*.pdf"))
    files = md_files + pdf_files

    if not files:
        logger.warning("No knowledge base files found in %s", abs_kb_dir)
        return {"files": 0, "chunks": 0}

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for filepath in files:
        filename = os.path.basename(filepath)
        topic = os.path.splitext(filename)[0]

        content = ""
        if filepath.lower().endswith(".pdf"):
            try:
                import PyPDF2
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            content += extracted + "\n"
                logger.info("Extracted %d chars from PDF: %s", len(content), filename)
            except ImportError:
                logger.warning("PyPDF2 not installed — skipping PDF %s", filename)
                continue
            except Exception as e:
                logger.error("Error reading PDF %s: %s", filename, e)
                continue
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

        if not content.strip():
            continue

        chunks = _chunk_text(content)
        logger.info("File '%s' → %d chunks", filename, len(chunks))

        for i, chunk in enumerate(chunks):
            chunk_id = f"file_{topic}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(
                {"source": filename, "topic": topic, "tags": topic, "chunk_index": i}
            )

    if all_chunks:
        collection.upsert(ids=all_ids, documents=all_chunks, metadatas=all_metadatas)

    logger.info(
        "Ingested %d file-based chunks from %d files into 'math_knowledge'.",
        len(all_chunks), len(files),
    )
    return {"files": len(files), "chunks": len(all_chunks), "collection": "math_knowledge"}


if __name__ == "__main__":
    print("Building structured JEE knowledge base...")
    r1 = build_jee_knowledge_base()
    print(f"  Structured chunks: {r1['chunks']}")

    print("Ingesting knowledge base files...")
    r2 = ingest_knowledge_base()
    print(f"  File-based chunks: {r2['chunks']} from {r2['files']} files")

    total = r1["chunks"] + r2["chunks"]
    print(f"✅ Total chunks in 'math_knowledge': {total}")

