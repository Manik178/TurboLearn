import os
import re
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from PyPDF2 import PdfReader
from tqdm import tqdm

# Vector DB
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
import weaviate.classes.init as wvc

# Embeddings
from sentence_transformers import SentenceTransformer


# -------------------------------
# Data Structure
# -------------------------------
@dataclass
class Section:
    book_slug: str
    book_title: str
    section_id: str
    text: str

from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# PDF Text Extraction
# -------------------------------
def extract_pdf_text(pdf_path: str, skip_pages: int = 0) -> str:
    logging.info(f"📖 Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    
    if skip_pages > 0:
        logging.info(f"⏩ Skipping the first {skip_pages} pages.")
        pages_to_process = reader.pages[skip_pages:]
    else:
        pages_to_process = reader.pages
    
    pages = []
    for page in pages_to_process:
        content = page.extract_text()
        if content:
            pages.append(content)
            
    text = "\n".join(pages)
    logging.info(f"✅ Extracted {len(text.split())} words from {os.path.basename(pdf_path)}")
    return text


# -------------------------------
# Section Splitting with Fallback
# -------------------------------
def split_into_sections(text: str, book_slug: str, book_title: str) -> List[Section]:
    logging.info("🔍 Splitting text into sections (with fallback)...")

    lines = text.split("\n")
    sections = []
    buffer = []
    section_id = 0
    current_title = ""

    for line in lines:
        line_stripped = line.strip()
        # Detect "Chapter X" OR numbered headings like "1.1"
        if re.match(r"^(Chapter\s+\d+|[0-9]+(\.[0-9]+)*)\s+", line_stripped, re.IGNORECASE):
            if buffer and current_title:
                section_id += 1
                sections.append(Section(
                    book_slug=book_slug,
                    book_title=book_title,
                    section_id=f"{book_slug}-sec{section_id}",
                    text=current_title + "\n" + "\n".join(buffer)
                ))
                buffer = []
            current_title = line_stripped
        else:
            buffer.append(line_stripped)

    if buffer and current_title:
        section_id += 1
        sections.append(Section(
            book_slug=book_slug,
            book_title=book_title,
            section_id=f"{book_slug}-sec{section_id}",
            text=current_title + "\n" + "\n".join(buffer)
        ))

    # Fallback if no sections
    if not sections:
        logging.warning("⚠️ No structured sections found, falling back to block chunking...")
        words = text.split()
        block_size = 2000
        overlap = 200
        i = 0
        while i < len(words):
            j = min(len(words), i + block_size)
            chunk = " ".join(words[i:j])
            section_id += 1
            sections.append(Section(
                book_slug=book_slug,
                book_title=book_title,
                section_id=f"{book_slug}-block{section_id}",
                text=chunk
            ))
            if j == len(words): break
            i = j - overlap
        logging.info(f"✅ Fallback produced {len(sections)} blocks")
    else:
        logging.info(f"✅ Found {len(sections)} sections")

    return sections


# -------------------------------
# Chunking
# -------------------------------
def word_chunks(text: str, words_per_chunk: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + words_per_chunk)
        chunk = " ".join(words[i:j])
        if chunk.strip():
            chunks.append(chunk)
        if j == len(words): break
        i = j - overlap
        if i < 0: i = 0
    return chunks


# -------------------------------
# Embedding (MiniLM only)
# -------------------------------
class Embedder:
    def __init__(self):
        self._local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logging.info("✅ Using Local MiniLM Embeddings (384-dim)")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        vectors = self._local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)


# -------------------------------
# Weaviate Sink
# -------------------------------
class WeaviateSink:
    def __init__(self, collection: str):
        url = os.getenv("WCD_URL")
        key = os.getenv("WCD_API_KEY")
        if not url or not key:
            raise RuntimeError("Set WCD_URL and WCD_API_KEY in your .env file")

        logging.info(f"🌐 Connecting to Weaviate at {url.split('//')[1]}")
        
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(key),
            skip_init_checks=True,
            additional_config=wvc.AdditionalConfig(
                timeout=wvc.Timeout(init=60)
            )
        )

        self.collection_name = collection
        # Always recreate schema fresh for MiniLM (384)
        self.recreate_schema()

    def recreate_schema(self):
        if self.client.collections.exists(self.collection_name):
            logging.info(f"🗑️ Dropping old collection: {self.collection_name}")
            self.client.collections.delete(self.collection_name)

        logging.info(f"🛠️ Creating new collection: {self.collection_name}")
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(),
            properties=[
                Property(name="book_slug", data_type=DataType.TEXT),
                Property(name="book_title", data_type=DataType.TEXT),
                Property(name="section_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ]
        )



    def upsert(self, items: List[Dict], vectors: np.ndarray):
        col = self.client.collections.get(self.collection_name)
        try:
            with col.batch.dynamic() as batch:
                for obj, vec in zip(items, vectors):
                    batch.add_object(
                        properties=obj,
                        vector=vec
                    )
            logging.info(f"✅ Inserted {len(items)} chunks into Weaviate")
        except Exception as e:
            logging.error(f"❌ Batch insert failed: {e}", exc_info=True)

    def close(self):
        self.client.close()
        logging.info("🔒 Closed Weaviate connection")


# -------------------------------
# Pipeline Orchestration
# -------------------------------
def build_objects(sec: Section, chunks: List[str]) -> List[Dict]:
    return [{
        "book_slug": sec.book_slug,
        "book_title": sec.book_title,
        "section_id": sec.section_id,
        "chunk_id": f"{sec.section_id}-{idx}",
        "text": chunk,
    } for idx, chunk in enumerate(chunks)]


def ingest_pdf(
        pdf_path: str,
        sink: WeaviateSink,
        embedder: Embedder,
        batch_size: int = 64,
        skip_pages: int = 20,
        chunk_size: int = 280,
        chunk_overlap: int = 60,
    ):
    book_slug = os.path.splitext(os.path.basename(pdf_path))[0]
    book_title = book_slug.replace("-", " ").title()

    raw_text = extract_pdf_text(pdf_path, skip_pages=skip_pages)
    sections = split_into_sections(raw_text, book_slug, book_title)

    all_objs, all_texts = [], []
    for sec in sections:
        chunks = word_chunks(sec.text, words_per_chunk=chunk_size, overlap=chunk_overlap)
        objs = build_objects(sec, chunks)
        all_objs.extend(objs)
        all_texts.extend([o["text"] for o in objs])

    logging.info(f"📦 Total chunks to embed for {book_slug}: {len(all_objs)}")

    # Embed in batches
    all_vectors = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="🔄 Embedding batches"):
        batch_texts = all_texts[i:i+batch_size]
        batch_vectors = embedder.embed(batch_texts)
        all_vectors.extend(batch_vectors)

    # Upload in batches
    for i in tqdm(range(0, len(all_objs), batch_size), desc="📤 Uploading batches"):
        batch_objs = all_objs[i:i+batch_size]
        batch_vecs = all_vectors[i:i+batch_size]
        sink.upsert(batch_objs, batch_vecs)

    logging.info(f"✅ Finished ingesting {book_slug}")


# -------------------------------
# CLI
# -------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ap = argparse.ArgumentParser(description="OpenStax PDF → Weaviate Ingestion Pipeline (MiniLM only)")
    ap.add_argument("--folder", default="books", help="Folder with PDF files")
    ap.add_argument("--collection", default="OpenStaxMiniLM", help="Weaviate collection name")
    ap.add_argument("--skip-pages", type=int, default=20, help="Pages to skip from start")
    ap.add_argument("--chunk-size", type=int, default=280, help="Words per chunk")
    ap.add_argument("--chunk-overlap", type=int, default=60, help="Words overlap between chunks")
    args = ap.parse_args()

    try:
        embedder = Embedder()
        sink = WeaviateSink(collection=args.collection)

        pdfs = [f for f in os.listdir(args.folder) if f.endswith(".pdf")]
        if not pdfs:
            logging.warning(f"⚠️ No PDF files found in {args.folder}")
            return
            
        for fname in pdfs:
            
            ingest_pdf(
                os.path.join(args.folder, fname), 
                sink, 
                embedder,
                skip_pages=args.skip_pages,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if 'sink' in locals() and sink.client.is_connected():
            sink.close()


if __name__ == "__main__":
    main()
