import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import chromadb
from pypdf import apply_configuration
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPDF_MAX_OUTPUT_LENGTH = 350_000_000
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = PROJECT_ROOT / "ingestion_manifest.json"


# ============================================================
# Logging setup
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
LOG_FILE = LOGS_DIR / f"{timestamp}_log.txt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

file_handler = logging.FileHandler(
    LOG_FILE,
    encoding="utf-8",
)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.propagate = False

logger.info(f"Logging to file: {LOG_FILE}")


# ============================================================
# Runtime dependencies
# ============================================================

settings = DocIngestionSettings()

logger.info("Loading embedding model...")

embedding_model = HuggingFaceEmbedding()


# ============================================================
# Helper functions
# ============================================================

def get_doc_name(document) -> str:
    """
    Return the source PDF file name from LlamaIndex document metadata.
    """

    metadata = document.metadata or {}

    return (
        metadata.get("file_name")
        or Path(str(metadata.get("file_path", ""))).name
        or "unknown_file"
    )


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA-256 hash for a source PDF.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_manifest() -> dict:
    """
    Load the ingestion manifest if it exists.
    """

    if not MANIFEST_FILE.exists():

        logger.info(
            "No ingestion manifest found. "
            "A new manifest will be created."
        )

        return {}

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

        if not isinstance(manifest, dict):

            logger.warning(
                "Manifest format is invalid. "
                "Starting with empty manifest."
            )

            return {}

        logger.info(
            f"Loaded ingestion manifest with "
            f"{len(manifest)} entries."
        )

        return manifest

    except Exception:

        logger.exception(
            "Failed to load ingestion manifest."
        )

        raise


def save_manifest(manifest: dict) -> None:
    """
    Save the ingestion manifest safely.

    The manifest is written to a temporary file first
    and then replaced.
    """

    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    temp_file.replace(MANIFEST_FILE)

    logger.info(
        f"Ingestion manifest saved: {MANIFEST_FILE}"
    )


def get_file_record_count(
    collection,
    file_name: str,
) -> int:
    """
    Return the number of Chroma records
    belonging to one source PDF.
    """

    result = collection.get(
        where={
            "file_name": file_name
        }
    )

    return len(
        result.get(
            "ids",
            []
        )
    )


def delete_file_records(
    collection,
    file_name: str,
) -> int:
    """
    Delete only the Chroma records belonging
    to one source PDF.

    The deletion is verified afterwards.
    """

    before_count = get_file_record_count(
        collection,
        file_name,
    )

    if before_count == 0:

        logger.info(
            f"[{file_name}] "
            "No existing Chroma records found. "
            "Nothing to delete."
        )

        return 0

    logger.info(
        f"[{file_name}] "
        f"Deleting {before_count} existing Chroma records..."
    )

    collection.delete(
        where={
            "file_name": file_name
        }
    )

    after_count = get_file_record_count(
        collection,
        file_name,
    )

    if after_count != 0:

        raise RuntimeError(
            f"[{file_name}] "
            "Chroma deletion verification failed. "
            f"{after_count} records still remain."
        )

    logger.info(
        f"[{file_name}] "
        f"Deleted {before_count} existing Chroma records."
    )

    return before_count


def determine_status(
    file_name: str,
    source_hash: str,
    collection,
    manifest: dict,
) -> tuple[str, int, str | None]:
    """
    Determine the incremental ingestion state
    for ONE source PDF.

    Returns:

        status
        Chroma record count
        stored hash

    Possible statuses:

        PENDING
        ADOPTED
        CURRENT
        OUTDATED
    """

    chroma_count = get_file_record_count(
        collection,
        file_name,
    )

    manifest_entry = manifest.get(
        file_name
    )

    stored_hash = None

    if isinstance(
        manifest_entry,
        dict,
    ):

        stored_hash = manifest_entry.get(
            "source_hash"
        )

    # --------------------------------------------------------
    # PENDING
    # --------------------------------------------------------

    if chroma_count == 0:

        return (
            "PENDING",
            chroma_count,
            stored_hash,
        )

    # --------------------------------------------------------
    # ADOPTED
    #
    # Existing vectors were created before the
    # SHA-256 manifest was introduced.
    #
    # Do NOT re-ingest them.
    # Simply record the current hash.
    # --------------------------------------------------------

    if not stored_hash:

        return (
            "ADOPTED",
            chroma_count,
            stored_hash,
        )

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    if stored_hash == source_hash:

        return (
            "CURRENT",
            chroma_count,
            stored_hash,
        )

    # --------------------------------------------------------
    # OUTDATED
    # --------------------------------------------------------

    return (
        "OUTDATED",
        chroma_count,
        stored_hash,
    )


def ingest_documents_for_file(
    documents,
    parser,
    vector_store,
    storage_context,
    file_name: str,
) -> int:
    """
    Parse and ingest ALL LlamaIndex documents/pages
    belonging to ONE source PDF.

    This is the critical function that prevents the
    previous page-by-page ingestion problem.
    """

    logger.info(
        f"[{file_name}] "
        f"Parsing {len(documents)} "
        "LlamaIndex documents/pages..."
    )

    nodes = parser.get_nodes_from_documents(
        documents
    )

    logger.info(
        f"[{file_name}] "
        f"Parsed nodes: {len(nodes)}"
    )

    if not nodes:

        raise RuntimeError(
            f"[{file_name}] "
            "No nodes were generated from the PDF."
        )

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embedding_model,
    )

    logger.info(
        f"[{file_name}] "
        "All nodes ingested successfully."
    )

    return len(nodes)


def get_source_pdf_files(
    docs_dir_path: Path,
) -> dict[str, Path]:
    """
    Discover actual PDF files in docs_dir.

    Returns:

        {
            "file_name.pdf": Path(...)
        }

    This lets us independently verify the number
    of physical source PDFs.
    """

    pdf_files = {}

    for pdf_path in docs_dir_path.glob("*.pdf"):

        pdf_files[pdf_path.name] = pdf_path

    return pdf_files


# ============================================================
# Main incremental ingestion
# ============================================================

def build_vector_store_from_documents():
    """
    Incrementally build/update the Chroma vector store.

    Processing is performed ONCE per source PDF.

    Rules:

        PENDING
            No Chroma vectors exist.
            -> Ingest ALL pages/documents for the PDF.

        ADOPTED
            Existing vectors exist but no SHA-256
            was previously stored.
            -> Record current hash.
            -> Do NOT re-ingest.

        CURRENT
            Existing vectors exist and source hash
            has not changed.
            -> Skip.

        OUTDATED
            Existing vectors exist but source hash
            has changed.
            -> Delete ALL existing vectors for PDF.
            -> Re-ingest ALL pages/documents.
            -> Update manifest only after successful ingestion.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "INCREMENTAL PDF INGESTION START"
    )

    logger.info(
        "============================================================"
    )

    try:

        # ====================================================
        # Configuration
        # ====================================================

        docs_dir_path = Path(
            settings.DOCUMENTS_DIR
        )

        vector_store_path = (
            settings.VECTOR_STORE_DIR
        )

        collection_name = (
            settings.COLLECTION_NAME
        )

        if not docs_dir_path.exists():

            raise FileNotFoundError(
                "Documents directory does not exist: "
                f"{docs_dir_path}"
            )

        # ====================================================
        # Load manifest
        # ====================================================

        manifest = load_manifest()

        # ====================================================
        # Discover actual PDF files
        # ====================================================

        source_pdf_files = get_source_pdf_files(
            docs_dir_path
        )

        logger.info(
            f"Source PDF files discovered: "
            f"{len(source_pdf_files)}"
        )

        if not source_pdf_files:

            raise RuntimeError(
                f"No PDF files found in {docs_dir_path}"
            )

        # ====================================================
        # Initialize Chroma
        # ====================================================

        logger.info(
            "Initializing ChromaDB persistent client at: "
            f"{vector_store_path}"
        )

        db = chromadb.PersistentClient(
            path=vector_store_path
        )

        collection = db.get_or_create_collection(
            name=collection_name
        )

        logger.info(
            f"Using Chroma collection: "
            f"{collection_name}"
        )

        initial_record_count = (
            collection.count()
        )

        logger.info(
            "Initial Chroma record count: "
            f"{initial_record_count}"
        )

        # ====================================================
        # Initialize parser
        # ====================================================

        parser = SimpleNodeParser(
            chunk_size=1024,
            chunk_overlap=50,
        )

        # ====================================================
        # Initialize vector store
        # ====================================================

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store
            )
        )

        # ====================================================
        # Load source documents/pages
        # ====================================================

        logger.info(
            "Loading documents from directory: "
            f"{docs_dir_path}"
        )

        logger.info(
            "Using pypdf maximum decompressed output length: "
            f"{PYPDF_MAX_OUTPUT_LENGTH:,} bytes"
        )

        loader = SimpleDirectoryReader(
            str(docs_dir_path)
        )

        with apply_configuration(
            zlib_maximum_output_length=PYPDF_MAX_OUTPUT_LENGTH
        ):
            documents = loader.load_data()

        logger.info(
            "Total LlamaIndex documents/pages loaded: "
            f"{len(documents)}"
        )

        # ====================================================
        # Group LlamaIndex documents by source PDF
        #
        # IMPORTANT:
        #
        # A single PDF may produce many LlamaIndex
        # Document objects, normally one per page.
        #
        # We must process the PDF as ONE unit.
        # ====================================================

        documents_by_file = {}

        for document in documents:

            file_name = get_doc_name(
                document
            )

            documents_by_file.setdefault(
                file_name,
                []
            ).append(
                document
            )

        logger.info(
            "Unique PDFs represented by "
            "LlamaIndex documents: "
            f"{len(documents_by_file)}"
        )

        # ====================================================
        # Compare physical PDFs with loader results
        # ====================================================

        source_pdf_names = set(
            source_pdf_files.keys()
        )

        loaded_pdf_names = set(
            documents_by_file.keys()
        )

        missing_from_loader = (
            source_pdf_names
            - loaded_pdf_names
        )

        unexpected_loaded_files = (
            loaded_pdf_names
            - source_pdf_names
        )

        if missing_from_loader:

            logger.error(
                "The following source PDFs were not "
                "returned by LlamaIndex:"
            )

            for file_name in sorted(
                missing_from_loader
            ):

                logger.error(
                    f"  - {file_name}"
                )

        if unexpected_loaded_files:

            logger.warning(
                "The following loaded files were not "
                "found in docs_dir:"
            )

            for file_name in sorted(
                unexpected_loaded_files
            ):

                logger.warning(
                    f"  - {file_name}"
                )

        # ====================================================
        # Counters
        #
        # These counters are PDF-level counters,
        # NOT page/document counters.
        # ====================================================

        adopted_files = []
        current_files = []
        pending_files = []
        updated_files = []
        failed_files = []

        total_nodes_created = 0

        # ====================================================
        # Process each UNIQUE source PDF exactly once
        # ====================================================

        for idx, file_name in enumerate(
            sorted(documents_by_file.keys()),
            start=1,
        ):

            file_documents = (
                documents_by_file[file_name]
            )

            logger.info(
                "------------------------------------------------------------"
            )

            logger.info(
                f"[{idx}/{len(documents_by_file)}] "
                f"Processing PDF: {file_name}"
            )

            logger.info(
                f"[{file_name}] "
                f"LlamaIndex documents/pages: "
                f"{len(file_documents)}"
            )

            try:

                # ------------------------------------------------
                # Get source path
                # ------------------------------------------------

                first_document = (
                    file_documents[0]
                )

                file_path_value = (
                    first_document.metadata.get(
                        "file_path"
                    )
                    if first_document.metadata
                    else None
                )

                if not file_path_value:

                    raise RuntimeError(
                        f"[{file_name}] "
                        "Source file path is missing."
                    )

                source_path = Path(
                    str(file_path_value)
                )

                if not source_path.exists():

                    raise FileNotFoundError(
                        f"[{file_name}] "
                        "Source file does not exist: "
                        f"{source_path}"
                    )

                # ------------------------------------------------
                # Calculate source hash
                # ------------------------------------------------

                source_hash = calculate_sha256(
                    source_path
                )

                # ------------------------------------------------
                # Determine PDF status
                # ------------------------------------------------

                status, chroma_count, stored_hash = (
                    determine_status(
                        file_name=file_name,
                        source_hash=source_hash,
                        collection=collection,
                        manifest=manifest,
                    )
                )

                logger.info(
                    f"[{file_name}] Status: {status}"
                )

                logger.info(
                    f"[{file_name}] "
                    f"Chroma records: {chroma_count}"
                )

                logger.info(
                    f"[{file_name}] "
                    f"Stored hash: {stored_hash}"
                )

                logger.info(
                    f"[{file_name}] "
                    f"Current hash: {source_hash}"
                )

                # =================================================
                # ADOPTED
                # =================================================

                if status == "ADOPTED":

                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "ADOPTED",
                        "recorded_at": (
                            datetime.now()
                            .astimezone()
                            .isoformat()
                        ),
                    }

                    adopted_files.append(
                        file_name
                    )

                    logger.info(
                        f"[{file_name}] "
                        "Existing vectors adopted as baseline. "
                        "No re-ingestion performed."
                    )

                # =================================================
                # CURRENT
                # =================================================

                elif status == "CURRENT":

                    current_files.append(
                        file_name
                    )

                    logger.info(
                        f"[{file_name}] "
                        "Source unchanged. "
                        "No ingestion required."
                    )

                # =================================================
                # PENDING
                # =================================================

                elif status == "PENDING":

                    logger.info(
                        f"[{file_name}] "
                        "New PDF detected. "
                        "Ingesting ALL pages/documents."
                    )

                    nodes_created = (
                        ingest_documents_for_file(
                            documents=file_documents,
                            parser=parser,
                            vector_store=vector_store,
                            storage_context=storage_context,
                            file_name=file_name,
                        )
                    )

                    total_nodes_created += (
                        nodes_created
                    )

                    new_record_count = (
                        get_file_record_count(
                            collection,
                            file_name,
                        )
                    )

                    if new_record_count == 0:

                        raise RuntimeError(
                            f"[{file_name}] "
                            "Ingestion completed but "
                            "no Chroma records were found."
                        )

                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "INGESTED",
                        "recorded_at": (
                            datetime.now()
                            .astimezone()
                            .isoformat()
                        ),
                    }

                    pending_files.append(
                        file_name
                    )

                    logger.info(
                        f"[{file_name}] "
                        "New PDF ingested successfully."
                    )

                    logger.info(
                        f"[{file_name}] "
                        f"Nodes created: {nodes_created}"
                    )

                    logger.info(
                        f"[{file_name}] "
                        f"Chroma records: {new_record_count}"
                    )

                # =================================================
                # OUTDATED
                # =================================================

                elif status == "OUTDATED":

                    logger.info(
                        f"[{file_name}] "
                        "Source content changed. "
                        "Replacing ALL existing vectors."
                    )

                    deleted_count = (
                        delete_file_records(
                            collection=collection,
                            file_name=file_name,
                        )
                    )

                    logger.info(
                        f"[{file_name}] "
                        f"Deleted {deleted_count} "
                        "old Chroma records."
                    )

                    nodes_created = (
                        ingest_documents_for_file(
                            documents=file_documents,
                            parser=parser,
                            vector_store=vector_store,
                            storage_context=storage_context,
                            file_name=file_name,
                        )
                    )

                    total_nodes_created += (
                        nodes_created
                    )

                    new_record_count = (
                        get_file_record_count(
                            collection,
                            file_name,
                        )
                    )

                    if new_record_count == 0:

                        raise RuntimeError(
                            f"[{file_name}] "
                            "Re-ingestion completed but "
                            "no Chroma records were found."
                        )

                    manifest[file_name] = {
                        "source_hash": source_hash,
                        "status": "UPDATED",
                        "recorded_at": (
                            datetime.now()
                            .astimezone()
                            .isoformat()
                        ),
                    }

                    updated_files.append(
                        file_name
                    )

                    logger.info(
                        f"[{file_name}] "
                        "Updated successfully."
                    )

                    logger.info(
                        f"[{file_name}] "
                        f"New nodes created: {nodes_created}"
                    )

                    logger.info(
                        f"[{file_name}] "
                        f"New Chroma records: "
                        f"{new_record_count}"
                    )

                # =================================================
                # UNKNOWN
                # =================================================

                else:

                    raise RuntimeError(
                        f"[{file_name}] "
                        f"Unknown ingestion status: "
                        f"{status}"
                    )

            except Exception as file_error:

                failed_files.append(
                    (
                        file_name,
                        str(file_error),
                    )
                )

                logger.exception(
                    f"[{file_name}] "
                    "Failed during incremental ingestion."
                )

        # ====================================================
        # Report source PDFs that LlamaIndex could not load
        # ====================================================

        for file_name in sorted(
            missing_from_loader
        ):

            failed_files.append(
                (
                    file_name,
                    "Source PDF exists in docs_dir "
                    "but was not returned by "
                    "SimpleDirectoryReader.",
                )
            )

        # ====================================================
        # Save manifest
        #
        # Only successful/adopted/current files should
        # have their existing manifest entries preserved.
        #
        # Failed files are NOT added as successful.
        # ====================================================

        save_manifest(
            manifest
        )

        # ====================================================
        # Final Chroma count
        # ====================================================

        final_record_count = (
            collection.count()
        )

        # ====================================================
        # Final verification
        # ====================================================

        manifest_pdf_names = set(
            manifest.keys()
        )

        missing_from_manifest = (
            source_pdf_names
            - manifest_pdf_names
        )

        extra_in_manifest = (
            manifest_pdf_names
            - source_pdf_names
        )

        logger.info(
            "============================================================"
        )

        logger.info(
            "INCREMENTAL INGESTION SUMMARY"
        )

        logger.info(
            "============================================================"
        )

        logger.info(
            f"Source PDF files discovered : "
            f"{len(source_pdf_files)}"
        )

        logger.info(
            f"LlamaIndex documents/pages  : "
            f"{len(documents)}"
        )

        logger.info(
            f"Unique PDFs loaded          : "
            f"{len(documents_by_file)}"
        )

        logger.info(
            f"Initial Chroma records      : "
            f"{initial_record_count}"
        )

        logger.info(
            f"Final Chroma records        : "
            f"{final_record_count}"
        )

        logger.info(
            f"ADOPTED PDFs                : "
            f"{len(adopted_files)}"
        )

        logger.info(
            f"CURRENT PDFs                : "
            f"{len(current_files)}"
        )

        logger.info(
            f"PENDING/NEW PDFs            : "
            f"{len(pending_files)}"
        )

        logger.info(
            f"UPDATED PDFs               : "
            f"{len(updated_files)}"
        )

        logger.info(
            f"FAILED PDFs                : "
            f"{len(failed_files)}"
        )

        logger.info(
            f"Nodes created              : "
            f"{total_nodes_created}"
        )

        logger.info(
            f"Manifest entries           : "
            f"{len(manifest)}"
        )

        logger.info(
            f"PDFs missing from manifest: "
            f"{len(missing_from_manifest)}"
        )

        logger.info(
            f"Extra manifest entries    : "
            f"{len(extra_in_manifest)}"
        )

        # ====================================================
        # Detailed file lists
        # ====================================================

        if adopted_files:

            logger.info(
                f"ADOPTED PDF names: "
                f"{sorted(adopted_files)}"
            )

        if current_files:

            logger.info(
                f"CURRENT PDF names: "
                f"{sorted(current_files)}"
            )

        if pending_files:

            logger.info(
                f"NEW PDF names: "
                f"{sorted(pending_files)}"
            )

        if updated_files:

            logger.info(
                f"UPDATED PDF names: "
                f"{sorted(updated_files)}"
            )

        # ====================================================
        # Missing manifest entries
        # ====================================================

        if missing_from_manifest:

            logger.error(
                "PDFs missing from manifest:"
            )

            for file_name in sorted(
                missing_from_manifest
            ):

                logger.error(
                    f"  - {file_name}"
                )

        # ====================================================
        # Extra manifest entries
        # ====================================================

        if extra_in_manifest:

            logger.warning(
                "Manifest entries not found in docs_dir:"
            )

            for file_name in sorted(
                extra_in_manifest
            ):

                logger.warning(
                    f"  - {file_name}"
                )

        # ====================================================
        # Failures
        # ====================================================

        if failed_files:

            logger.error(
                "Failure reasons:"
            )

            for file_name, reason in failed_files:

                logger.error(
                    f"- {file_name}: {reason}"
                )

        # ====================================================
        # Final success/failure determination
        # ====================================================

        if failed_files:

            logger.error(
                "============================================================"
            )

            logger.error(
                "INGESTION COMPLETED WITH FAILURES"
            )

            logger.error(
                "============================================================"
            )

            return 1

        if missing_from_manifest:

            logger.error(
                "============================================================"
            )

            logger.error(
                "INGESTION COMPLETED BUT MANIFEST IS INCOMPLETE"
            )

            logger.error(
                "============================================================"
            )

            return 1

        logger.info(
            "============================================================"
        )

        logger.info(
            "INCREMENTAL INGESTION COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "============================================================"
        )

        return 0

    except Exception:

        logger.exception(
            "Fatal error occurred during incremental ingestion."
        )

        return 1


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        build_vector_store_from_documents()
    )