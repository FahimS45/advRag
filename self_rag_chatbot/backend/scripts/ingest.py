"""
One-time document ingestion script.

Usage
-----
  # From the project root:
  python -m scripts.ingest

  # Or directly:
  python scripts/ingest.py

Place your company PDF files in the `./documents/` directory before running.
The script is idempotent: re-running it will add new chunks without dropping
existing ones (use --reset to wipe and reload everything).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 150


def load_pdfs(directory: Path) -> list:
    docs = []
    pdf_files = sorted(directory.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return docs
    for pdf_path in pdf_files:
        logger.info("Loading %s …", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Tag each chunk with its source filename
        for page in pages:
            page.metadata.setdefault("source", pdf_path.name)
        docs.extend(pages)
        logger.info("  → %d page(s)", len(pages))
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks


def build_vectorstore(reset: bool) -> PGVector:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    vs = PGVector(
        embeddings=embeddings,
        collection_name=settings.VECTOR_TABLE_NAME,
        connection=settings.VECTOR_DB_URI,
        use_jsonb=True,
        pre_delete_collection=reset,   # wipe before reload when --reset is passed
    )
    return vs


def ingest(reset: bool = False) -> None:
    logger.info("Ingestion starting (reset=%s) …", reset)

    raw_docs = load_pdfs(DOCUMENTS_DIR)
    if not raw_docs:
        logger.error("Nothing to ingest. Aborting.")
        sys.exit(1)

    chunks = chunk_documents(raw_docs)

    logger.info("Connecting to vector store …")
    vs = build_vectorstore(reset=reset)

    logger.info("Embedding and storing %d chunks …", len(chunks))
    vs.add_documents(chunks)

    logger.info("✓ Ingestion complete. %d documents stored in '%s'.",
                len(chunks), settings.VECTOR_TABLE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into PGVector.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before re-ingesting.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
