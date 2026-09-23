import logging
from datetime import datetime
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings

# -----------------------------
# Logging setup (one file per run)
# -----------------------------
# Resolve project root from this file location: src/rag_doc_ingestion/ingest_docs.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Keep all ingestion logs in a dedicated folder under project root.
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create one log file per run: YYYYMMDDHHMMSS_log.txt
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
LOG_FILE = LOGS_DIR / f"{timestamp}_log.txt"

# Configure logger to write to both terminal and file.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Clear existing handlers so repeated runs do not duplicate log lines.
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# File handler writes persistent logs for troubleshooting and audits.
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

# Console handler keeps real-time feedback in terminal.
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
# Disable propagation to root logger to avoid duplicate console logs.
logger.propagate = False
logger.info(f"Logging to file: {LOG_FILE}")

# -----------------------------
# Runtime dependencies
# -----------------------------
# Load ingestion settings from .env via DocIngestionSettings.
settings = DocIngestionSettings()

# Download/load embedding model once and reuse it for all documents.
logger.info("Loading embedding model...")
embedding_model = HuggingFaceEmbedding()


def get_doc_name(document) -> str:
    """Return a readable file name from LlamaIndex document metadata."""
    metadata = document.metadata or {}
    return (
        # Some loaders provide direct file_name.
        metadata.get("file_name")
        # Others provide file_path; extract basename in that case.
        or Path(str(metadata.get("file_path", ""))).name
        # Final fallback when metadata is incomplete.
        or "unknown_file"
    )

def build_vector_store_from_documents():
    """Read documents, index them into Chroma, and log per-file outcomes."""
    logger.info("Building vector store from documents...")

    try:
        # Use uppercase setting names that match DocIngestionSettings fields.
        docs_dir_path = settings.DOCUMENTS_DIR
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        # Load all supported files from the source directory.
        logger.info(f"Loading documents from directory: {docs_dir_path}")
        loader = SimpleDirectoryReader(docs_dir_path)
        documents = loader.load_data()
        total_files = len(documents)
        logger.info(f"Total files discovered: {total_files}")

        # Initialize parser and vector store once, then ingest each file separately.
        # Per-file ingestion helps us identify exact failures and pending files.
        parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=50)
        logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
        db = chromadb.PersistentClient(path=vector_store_path)
        collection = db.get_or_create_collection(name=collection_name)
        logger.info(f"Using Chroma collection: {collection_name}")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Track per-file ingestion status for final summary logs.
        succeeded_files = []
        failed_files = []

        for idx, doc in enumerate(documents, start=1):
            file_name = get_doc_name(doc)
            logger.info(f"[{idx}/{total_files}] Processing file: {file_name}")

            try:
                # Parse only this file's content into vectorizable chunks.
                nodes = parser.get_nodes_from_documents([doc])
                logger.info(f"[{file_name}] Parsed nodes: {len(nodes)}")

                # Create/update vector index for the current file.
                VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
                succeeded_files.append(file_name)
                logger.info(f"[{file_name}] Ingested successfully")
            except Exception as file_error:
                # Capture file-level reason and full traceback in logs.
                failed_files.append((file_name, str(file_error)))
                logger.exception(f"[{file_name}] Failed during ingestion")

        # Pending files are the ones that failed in this run.
        pending_files = [name for name, _ in failed_files]

        # Final run summary to quickly audit ingestion state.
        logger.info("========== INGESTION SUMMARY ==========")
        logger.info(f"Total files discovered : {total_files}")
        logger.info(f"Successfully ingested  : {len(succeeded_files)}")
        logger.info(f"Failed/Pending files   : {len(pending_files)}")
        logger.info(f"Successful file names  : {succeeded_files}")
        logger.info(f"Pending file names     : {pending_files}")

        if failed_files:
            logger.info("Failure reasons by file:")
            for file_name, reason in failed_files:
                logger.info(f"- {file_name}: {reason}")

        logger.info("Vector store build completed")
        # Exit code: 0 when all files ingested, 1 when any file failed.
        return 0 if not failed_files else 1
    

    except Exception:
        # Top-level fallback for setup/runtime failures outside file loop.
        logger.exception("Error occurred while building vector store")
        return 1


if __name__ == "__main__":
    raise SystemExit(build_vector_store_from_documents())
