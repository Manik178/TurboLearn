import os
import argparse
import logging
import textwrap
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
load_dotenv()
# --- Weaviate ---
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# --- Embeddings for QUERY ONLY (must match 384-dim in your collection) ---
from sentence_transformers import SentenceTransformer

# --- LLM for lesson drafting (generation only) ---
import google.generativeai as genai

# --- PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -----------------------
# ENV expected:
#   WCD_URL, WCD_API_KEY  -> your Weaviate Cloud cluster
#   GOOGLE_API_KEY        -> for Gemini (generation only)
# -----------------------

# ============== CONNECTIONS ==============

def connect_weaviate() -> weaviate.WeaviateClient:
    url = os.getenv("WCD_URL")
    key = os.getenv("WCD_API_KEY")
    if not url or not key:
        raise RuntimeError("Set WCD_URL and WCD_API_KEY in your environment/.env")

    logging.info(f"🌐 Connecting to Weaviate @ {url.split('//')[1]}")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
        # skip gRPC init health checks to avoid transient failures
        skip_init_checks=True,
    )
    return client


def init_genai():
    gkey = os.getenv("GOOGLE_API_KEY")
    if not gkey:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment/.env for generation.")
    genai.configure(api_key=gkey)

# ============== EMBEDDING (QUERY) ==============

from typing import List, Dict, Optional
import numpy as np
from weaviate.classes.query import MetadataQuery, Filter

# Make sure your query embedder returns 384-d vectors (MiniLM) to match the collection
class QueryEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], convert_to_numpy=True)[0]  # shape (384,)
        return vec.astype(np.float32)

def make_filters(
    book_title: Optional[str] = None,
    book_slug: Optional[str] = None,
    section_id_prefix: Optional[str] = None,
):
    clauses = []
    if book_title:
        clauses.append(Filter.by_property("book_title").equal(book_title))
    if book_slug:
        clauses.append(Filter.by_property("book_slug").equal(book_slug))
    if section_id_prefix:
        clauses.append(Filter.by_property("section_id").like(f"{section_id_prefix}*"))
    if not clauses:
        return None
    # AND them together
    f = clauses[0]
    for c in clauses[1:]:
        f = f & c
    return f

def semantic_search(
    client,
    collection: str,
    query_text: str,
    top_k: int = 12,
    book_title: Optional[str] = None,
    book_slug: Optional[str] = None,
    section_id_prefix: Optional[str] = None,
) -> List[Dict]:
    """
    Returns a list of dicts with: text, book_slug, book_title, section_id, chunk_id, distance
    Uses MiniLM (384-d) at query time to match your OpenStaxMiniLM collection.
    """
    # 1) Embed the query (MiniLM: 384-d — matches your collection)
    vec = QueryEmbedder().embed(query_text)

    # 2) Optional filters
    filters = make_filters(book_title=book_title, book_slug=book_slug, section_id_prefix=section_id_prefix)

    # 3) Execute the vector search (no .fetch() in v4)
    col = client.collections.get(collection)
    res = col.query.near_vector(
        near_vector=vec.tolist(),
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
        return_properties=["text", "book_slug", "book_title", "section_id", "chunk_id"],
        filters=filters,
    )

    # 4) Parse results (metadata is an object, not a dict)
    out = []
    for obj in res.objects or []:
        props = obj.properties or {}
        dist = obj.metadata.distance if getattr(obj, "metadata", None) else None
        out.append({
            "text": props.get("text", ""),
            "book_slug": props.get("book_slug", ""),
            "book_title": props.get("book_title", ""),
            "section_id": props.get("section_id", ""),
            "chunk_id": props.get("chunk_id", ""),
            "distance": dist,
        })
    return out
def build_context(snippets: List[Dict], max_chars: int = 8000) -> str:
    """
    Concatenate top chunks into a single context string with lightweight headings.
    """
    parts = []
    running = 0
    for s in snippets:
        header = f"[{s['book_title']} / {s['section_id']}]"
        block = f"{header}\n{s['text']}\n"
        if running + len(block) > max_chars:
            break
        parts.append(block)
        running += len(block)
    return "\n\n".join(parts)

# ============== LESSON GENERATION (Gemini) ==============

def generate_lesson(topic: str, context: str, difficulty: str = "intermediate") -> str:
    """
    Uses Gemini for composing the final lesson from retrieved context.
    """
    init_genai()
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    prompt = f"""
You are an expert teacher. Create an adaptive lesson on the topic:
"{topic}"

Target level: {difficulty}

Use ONLY the context below for factual grounding. If something is not in context,
you may add general teaching glue text (explanations, analogies), but avoid introducing facts
that contradict the context.

=== CONTEXT START ===
{context}
=== CONTEXT END ===

Output in the following clear structure using markdown-like headings:

# Title: {topic} ({difficulty.capitalize()})

## Overview
- 3-6 short paragraphs to introduce and explain the key ideas tailored for {difficulty} learners.

## Key Concepts (Bulleted)
- 6-12 bullets, short and precise; include important formulas only if appropriate for {difficulty}.

## Worked Example(s)
- 1-3 worked examples with step-by-step reasoning.

## Practice Questions
- 8-12 questions of mixed difficulty suitable for {difficulty}, no answers here.

## Answers and Explanations
- Provide numbered answers with concise explanations.
"""
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") else str(resp)

# ============== PDF RENDERING (ReportLab) ==============

def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        name="H1",
        fontSize=18,
        leading=22,
        spaceBefore=12,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="H2",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontSize=11,
        leading=15,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="Mono",
        fontName="Courier",
        fontSize=10,
        leading=14,
        spaceAfter=6
    ))
    return styles


def _md_like_to_flowables(text: str, styles) -> List:
    """
    Very lightweight parser: lines starting with '#', '##', '-', digits, etc.
    Creates Paragraphs and bullet lists. Keeps code-style fences as mono.
    """
    lines = text.splitlines()
    story = []
    bullets = []
    in_code = False
    code_buffer = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(b, styles["Body"])) for b in bullets],
                bulletType="bullet",
                leftPadding=18
            ))
            bullets = []

    def flush_code():
        nonlocal code_buffer
        if code_buffer:
            block = "\n".join(code_buffer)
            story.append(Paragraph(
                "<br/>".join([textwrap.fill(l, 110) for l in block.split("\n")]),
                styles["Mono"]
            ))
            code_buffer = []

    for raw in lines:
        line = raw.rstrip()

        # simple code-fence
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
            else:
                in_code = False
                flush_code()
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(line[2:].strip(), styles["H1"]))
        elif line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(line[3:].strip(), styles["H2"]))
        elif line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.strip() == "":
            flush_bullets()
            story.append(Spacer(1, 0.12 * inch))
        else:
            flush_bullets()
            story.append(Paragraph(textwrap.fill(line, 110), styles["Body"]))

    flush_bullets()
    flush_code()
    return story


def make_pdf(title: str, lesson_text: str, out_path: str):
    styles = _styles()
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        topMargin=0.8*inch, bottomMargin=0.8*inch
    )

    story = []
    # Title page
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph(title, styles["TitleCenter"]))
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("Auto-generated lesson (context-grounded)", styles["Body"]))
    story.append(PageBreak())

    # Main content (md-like)
    story.extend(_md_like_to_flowables(lesson_text, styles))

    doc.build(story)
    logging.info(f"📄 Saved PDF: {out_path}")

# ============== ORCHESTRATION ==============

def run_pipeline(
    topic: str,
    difficulty: str,
    collection: str = "OpenStaxMiniLM",
    top_k: int = 12,
    output_dir: str = "outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    client = connect_weaviate()
    try:
        # 1) Retrieve grounded chunks with MiniLM query vector
        logging.info(f"🔎 Searching top-{top_k} chunks for topic: {topic}")
        hits = semantic_search(client, collection, topic, top_k=top_k)

        if not hits:
            raise RuntimeError("No results returned from Weaviate. Check collection name / data.")

        # 2) Build context window
        context = build_context(hits, max_chars=9000)

        # 3) Generate lesson with Gemini (uses GOOGLE_API_KEY)
        logging.info("🧠 Generating lesson with Gemini (grounded)...")
        lesson_text = generate_lesson(topic, context, difficulty=difficulty)

        # 4) Save PDF
        safe_topic = topic.strip().lower().replace(" ", "_").replace("/", "_")
        out_path = os.path.join(output_dir, f"{safe_topic}_{difficulty}.pdf")
        title = f"{topic} — {difficulty.capitalize()} Lesson"
        make_pdf(title, lesson_text, out_path)

        logging.info("✅ Done.")
        return out_path
    finally:
        if client.is_connected():
            client.close()

# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(
        description="Retrieval-augmented lesson → PDF (MiniLM for search, Gemini for generation)"
    )
    parser.add_argument("--topic", required=True, help="Topic to teach, e.g., 'Newton's Second Law'")
    parser.add_argument("--difficulty", default="intermediate",
                        choices=["beginner", "intermediate", "advanced"])
    parser.add_argument("--collection", default="OpenStaxMiniLM")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out = run_pipeline(
        topic=args.topic,
        difficulty=args.difficulty,
        collection=args.collection,
        top_k=args.top_k,
        output_dir=args.output_dir
    )
    print(f"\nPDF saved at: {out}\n")

# ============== STREAMLIT / WRAPPER ==============
def generate_pdf_lesson(topic: str, difficulty: str, output_dir: str = "outputs") -> str:
    """
    Wrapper for Streamlit or other apps.
    Runs the retrieval → generation → PDF pipeline and returns the saved file path.
    """
    return run_pipeline(
        topic=topic,
        difficulty=difficulty,
        collection="OpenStaxMiniLM",
        top_k=12,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
