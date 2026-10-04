import hashlib
import json
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
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = PROJECT_ROOT / "ingestion_manifest.json"


# -----------------------------
# Logging setup
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
LOG_FILE = LOGS_DIR / f"{timestamp}_log.txt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

logger.info(f"Logging to file: {LOG_FILE}")


# -----------------------------
# Runtime dependencies
# -----------------------------
settings = DocIngestionSettings()

logger.info("Loading embedding model...")
embedding_model = HuggingFaceEmbedding()


# -----------------------------
# Helper functions
# -----------------------------
def get_doc_name(document) -> str:
    """Return a readable file name from LlamaIndex document metadata."""
    metadata = document.metadata or {}

    return (
        metadata.get("file_name")
        or Path(str(metadata.get("file_path", ""))).name
        or "unknown_file"
    )


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash for a source document."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_manifest() -> dict:
    """Load the ingestion manifest if it exists."""
    if not MANIFEST_FILE.exists():
        logger.info("No ingestion manifest found. A new manifest will be created.")
        return {}

    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict):
            logger.warning("Manifest format is invalid. Starting with empty manifest.")
            return {}

        logger.info(
            f"Loaded ingestion manifest with {len(manifest)} entries."
        )

        return manifest

    except Exception:
        logger.exception("Failed to load ingestion manifest.")
        raise


def save_manifest(manifest: dict) -> None:
    """Save the ingestion manifest safely."""
    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    temp_file.replace(MANIFEST_FILE)

    logger.info(f"Ingestion manifest saved: {MANIFEST_FILE}")


def get_file_record_count(collection, file_name: str) -> int:
    """Return the number of Chroma records belonging to one source PDF."""
    result = collection.get(
        where={"file_name": file_name}
    )

    return len(result.get("ids", []))


def delete_file_records(collection, file_name: str) -> int:
    """Delete only the Chroma records belonging to one source PDF."""
    before_count = get_file_record_count(collection, file_name)

    if before_count == 0:
        logger.info(
            f"[{file_name}] No existing Chroma records found. Nothing to delete."
        )
        return 0

    collection.delete(
        where={"file_name": file_name}
    )

    after_count = get_file_record_count(collection, file_name)

    if after_count != 0:
        raise RuntimeError(
            f"[{file_name}] Chroma deletion verification failed. "
            f"{after_count} records still remain."
        )

    logger.info(
        f"[{file_name}] Deleted {before_count} existing Chroma records."
    )

    return before_count


def determine_status(
    file_name: str,
    source_hash: str,
    collection,
    manifest: dict,
) -> tuple[str, int, str | None]:
    """
    Determine the incremental ingestion state.

    Returns:
        status, Chroma record count, stored hash
    """
    chroma_count = get_file_record_count(collection, file_name)

    manifest_entry = manifest.get(file_name)

    stored_hash = None

    if isinstance(manifest_entry, dict):
        stored_hash = manifest_entry.get("source_hash")

    # No vectors exist for this PDF.
    if chroma_count == 0:
        return "PENDING", chroma_count, stored_hash

    # Existing vectors but no previous fingerprint.
    # Adopt current content as baseline without re-ingesting.
    if not stored_hash:
        return "ADOPTED", chroma_count, stored_hash

    # Existing vectors and unchanged source.
    if stored_hash == source_hash:
        return "CURRENT", chroma_count, stored_hash

    # Existing vectors but source content changed.
    return "OUTDATED", chroma_count, stored_hash


def ingest_single_document(
    document,
    parser,
    vector_store,
    storage_context,
    file_name: str,
) -> int:
    """Parse and ingest one document."""
    nodes = parser.get_nodes_from_documents([document])

    logger.info(
        f"[{file_name}] Parsed nodes: {len(nodes)}"
    )

    if not nodes:
        raise RuntimeError(
            f"[{file_name}] No nodes were generated from the document."
        )

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=embedding_model,
    )

    logger.info(
        f"[{file_name}] Ingested successfully."
    )

    return len(nodes)


# -----------------------------
# Main incremental ingestion
# -----------------------------
def build_vector_store_from_documents():
    """
    Incrementally build/update the Chroma vector store.

    Rules:
        PENDING  -> ingest
        ADOPTED  -> record current hash, do not ingest
        CURRENT  -> skip
        OUTDATED -> delete old vectors and re-ingest
    """
    logger.info("========== INCREMENTAL INGESTION START ==========")

    try:
        docs_dir_path = Path(settings.DOCUMENTS_DIR)
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        if not docs_dir_path.exists():
            raise FileNotFoundError(
                f"Documents directory does not exist: {docs_dir_path}"
            )

        # Load manifest.
        manifest = load_manifest()

        # Initialize Chroma.
        logger.info(
            f"Initializing ChromaDB persistent client at: {vector_store_path}"
        )

        db = chromadb.PersistentClient(
            path=vector_store_path
        )

        collection = db.get_or_create_collection(
            name=collection_name
        )

        logger.info(
            f"Using Chroma collection: {collection_name}"
        )

        initial_record_count = collection.count()

        logger.info(
            f"Initial Chroma record count: {initial_record_count}"
        )

        # Initialize parser/vector store.
        parser = SimpleNodeParser(
            chunk_size=1024,
            chunk_overlap=50,
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Load source documents.
        logger.info(
            f"Loading documents from directory: {docs_dir_path}"
        )

        loader = SimpleDirectoryReader(
            str(docs_dir_path)
        )

        documents = loader.load_data()

        logger.info(
            f"Documents discovered: {len(documents)}"
        )

        # Counters.
        adopted_files = []
        current_files = []
        pending_files = []
        updated_files = []
        failed_files = []

        total_nodes_created = 0

        # Process each source document.
        for idx, document in enumerate(
            documents,
            start=1,
        ):
            file_name = get_doc_name(document)

            logger.info(
                f"[{idx}/{len(documents)}] Checking file: {file_name}"
            )

            try:
                file_path_value = (
                    document.metadata.get("file_path")
                    if document.metadata
                    else None
                )

                if not file_path_value:
                    raise RuntimeError(
                        f"[{file_name}] Source file path is missing."
                    )

                source_path = Path(str(file_path_value))

                if not source_path.exists():
                    raise FileNotFoundError(
                        f"[{file_name}] Source file does not exist: {source_path}"
                    )

                source_hash = calculate_sha256(
                    source_path
                )

                status, chroma_count, stored_hash = determine_status(
                    file_name=file_name,
                    source_hash=source_hash,
                    collection=collection,
                    manifest=manifest,
                )

                logger.info(
                    f"[{file_name}] Status: {status}"
                )

                logger.info(
                    f"[{file_name}] Chroma records: {chroma_count}"
                )

                logger.info(
                    f"[{file_name}] Stored hash: {stored_hash}"
                )

                logger.info(
                    f"[{file_name}] Current hash: {source_hash}"
                )

                # -----------------------------
                # ADOPTED
                # -----------------------------
                if status == "ADOPTED":
                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "ADOPTED",
                        "recorded_at": datetime.now().astimezone().isoformat(),
                    }

                    adopted_files.append(file_name)

                    logger.info(
                        f"[{file_name}] Existing vectors adopted as baseline. "
                        "No re-ingestion performed."
                    )

                # -----------------------------
                # CURRENT
                # -----------------------------
                elif status == "CURRENT":
                    current_files.append(file_name)

                    logger.info(
                        f"[{file_name}] Source unchanged. "
                        "No ingestion required."
                    )

                # -----------------------------
                # PENDING
                # -----------------------------
                elif status == "PENDING":
                    nodes_created = ingest_single_document(
                        document=document,
                        parser=parser,
                        vector_store=vector_store,
                        storage_context=storage_context,
                        file_name=file_name,
                    )

                    total_nodes_created += nodes_created

                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "INGESTED",
                        "recorded_at": datetime.now().astimezone().isoformat(),
                    }

                    pending_files.append(file_name)

                    logger.info(
                        f"[{file_name}] New document ingested."
                    )

                # -----------------------------
                # OUTDATED
                # -----------------------------
                elif status == "OUTDATED":
                    logger.info(
                        f"[{file_name}] Source content changed. "
                        "Replacing existing vectors."
                    )

                    deleted_count = delete_file_records(
                        collection=collection,
                        file_name=file_name,
                    )

                    logger.info(
                        f"[{file_name}] Deleted {deleted_count} old vectors."
                    )

                    nodes_created = ingest_single_document(
                        document=document,
                        parser=parser,
                        vector_store=vector_store,
                        storage_context=storage_context,
                        file_name=file_name,
                    )

                    total_nodes_created += nodes_created

                    new_record_count = get_file_record_count(
                        collection,
                        file_name,
                    )

                    if new_record_count == 0:
                        raise RuntimeError(
                            f"[{file_name}] Re-ingestion completed but "
                            "no Chroma records were found."
                        )

                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "UPDATED",
                        "recorded_at": datetime.now().astimezone().isoformat(),
                    }

                    updated_files.append(file_name)

                    logger.info(
                        f"[{file_name}] Updated successfully. "
                        f"New Chroma records: {new_record_count}"
                    )

                else:
                    raise RuntimeError(
                        f"[{file_name}] Unknown ingestion status: {status}"
                    )

            except Exception as file_error:
                failed_files.append(
                    (file_name, str(file_error))
                )

                logger.exception(
                    f"[{file_name}] Failed during incremental ingestion."
                )

        # Save manifest only after processing all files.
        save_manifest(manifest)

        final_record_count = collection.count()

        # -----------------------------
        # Final summary
        # -----------------------------
        logger.info(
            "========== INCREMENTAL INGESTION SUMMARY =========="
        )

        logger.info(
            f"Initial Chroma records : {initial_record_count}"
        )

        logger.info(
            f"Final Chroma records   : {final_record_count}"
        )

        logger.info(
            f"ADOPTED files         : {len(adopted_files)}"
        )

        logger.info(
            f"CURRENT files         : {len(current_files)}"
        )

        logger.info(
            f"PENDING/NEW files     : {len(pending_files)}"
        )

        logger.info(
            f"UPDATED files         : {len(updated_files)}"
        )

        logger.info(
            f"FAILED files          : {len(failed_files)}"
        )

        logger.info(
            f"Nodes created         : {total_nodes_created}"
        )

        if adopted_files:
            logger.info(
                f"ADOPTED file names: {adopted_files}"
            )

        if pending_files:
            logger.info(
                f"NEW file names: {pending_files}"
            )

        if updated_files:
            logger.info(
                f"UPDATED file names: {updated_files}"
            )

        if failed_files:
            logger.info("Failure reasons:")

            for file_name, reason in failed_files:
                logger.info(
                    f"- {file_name}: {reason}"
                )

        logger.info(
            "========== INCREMENTAL INGESTION COMPLETED =========="
        )

        return 0 if not failed_files else 1

    except Exception:
        logger.exception(
            "Fatal error occurred during incremental ingestion."
        )

        return 1


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    raise SystemExit(
        build_vector_store_from_documents()
    )