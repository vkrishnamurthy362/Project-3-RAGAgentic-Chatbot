from collections import defaultdict
import csv
from pathlib import Path
from datetime import datetime
import re

import chromadb
from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings


# Timestamp prefix used by logging format in ingest_docs.py.
TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} ")


def _extract_file_name_from_metadata(metadata: dict | None) -> str:
    """Extract a readable source file name from Chroma/LlamaIndex metadata."""
    if not metadata:
        return "unknown_file"

    # Common keys used by loaders and vector pipelines.
    file_name = metadata.get("file_name") or metadata.get("filename")
    if file_name:
        return Path(str(file_name)).name

    source_path = metadata.get("file_path") or metadata.get("source")
    if source_path:
        return Path(str(source_path)).name

    return "unknown_file"


def _get_project_root() -> Path:
    """Resolve project root from this file path."""
    return Path(__file__).resolve().parents[2]


def _get_source_files(docs_dir: str) -> list[str]:
    """Read source folder and return all file names expected for ingestion."""
    docs_path = Path(docs_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        return []

    return sorted([p.name for p in docs_path.iterdir() if p.is_file()])


def _find_latest_log_file(project_root: Path) -> Path | None:
    """Return the latest ingestion log file from the logs folder, if any."""
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        return None

    log_files = sorted(logs_dir.glob("*_log.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


def _parse_failure_blocks_from_log(log_file: Path | None) -> dict[str, list[str]]:
    """Parse per-file failure traceback blocks from latest log output."""
    if log_file is None or not log_file.exists():
        return {}

    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    failures: dict[str, list[str]] = defaultdict(list)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect the file-level failure line created by logger.exception.
        marker = "Failed during ingestion"
        if marker in line and "[" in line and "]" in line:
            start = line.rfind("[")
            end = line.rfind("]")
            file_name = line[start + 1:end] if start != -1 and end != -1 and end > start else "unknown_file"

            # Capture current line + traceback lines until next timestamped log entry.
            block_lines = [line]
            j = i + 1
            while j < len(lines) and not TIMESTAMP_PREFIX_RE.match(lines[j]):
                block_lines.append(lines[j])
                j += 1

            failures[file_name].append("\n".join(block_lines))
            i = j
            continue

        i += 1

    return failures


def _build_ingested_file_stats(collection) -> dict[str, int]:
    """Return chunk counts per source file from Chroma metadata."""
    total_records = collection.count()
    if total_records == 0:
        return {}

    # Pull all metadata so we can aggregate by source file.
    data = collection.get(include=["metadatas"], limit=total_records)
    metadatas = data.get("metadatas", [])

    chunk_counts: dict[str, int] = defaultdict(int)
    for metadata in metadatas:
        file_name = _extract_file_name_from_metadata(metadata)
        chunk_counts[file_name] += 1

    return dict(chunk_counts)


def _get_latest_error_summary(blocks: list[str]) -> str:
    """Return the most relevant final error line from a traceback block list."""
    if not blocks:
        return ""

    latest_block = blocks[-1]
    lines = [line.strip() for line in latest_block.splitlines() if line.strip()]

    # Most Python tracebacks end with the concise exception line.
    for line in reversed(lines):
        if "Error" in line or "Exception" in line or "Traceback" in line:
            return line

    return lines[-1] if lines else ""


def _write_status_csv(
    logs_dir: Path,
    source_files: list[str],
    ingested_stats: dict[str, int],
    failures: dict[str, list[str]],
) -> Path:
    """Write a status report CSV with file-level ingestion and error summary details."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_path = logs_dir / f"{timestamp}_ingestion_status_report.csv"

    all_files = sorted(set(source_files) | set(ingested_stats.keys()) | set(failures.keys()))

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "status", "chunks", "has_error", "latest_error_summary"])

        for file_name in all_files:
            chunks = ingested_stats.get(file_name, 0)
            has_error = file_name in failures
            status = "INGESTED" if chunks > 0 else "PENDING/NOT_INGESTED"
            if has_error:
                status = "FAILED"

            latest_error_summary = _get_latest_error_summary(failures.get(file_name, []))
            writer.writerow([file_name, status, chunks, has_error, latest_error_summary])

    return csv_path


def main() -> None:
    # -----------------------------
    # Load runtime settings
    # -----------------------------
    settings = DocIngestionSettings()
    project_root = _get_project_root()
    logs_dir = project_root / "logs"

    # -----------------------------
    # Connect to Chroma and print overview
    # -----------------------------
    client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
    collections = client.list_collections()

    print("=== ChromaDB Overview ===")
    print(f"Vector store path: {settings.VECTOR_STORE_DIR}")
    print(f"Configured collection: {settings.COLLECTION_NAME}")
    print(f"Collections found: {len(collections)}")

    if not collections:
        print("No collections found. Run ingestion first.")
        return

    print("\nCollection names:")
    for collection_obj in collections:
        print(f"- {collection_obj.name}")

    collection = client.get_collection(settings.COLLECTION_NAME)
    total_records = collection.count()
    print(f"\nTotal records in '{settings.COLLECTION_NAME}': {total_records}")

    # -----------------------------
    # Build file status from DB metadata + source folder
    # -----------------------------
    source_files = _get_source_files(settings.DOCUMENTS_DIR)
    ingested_stats = _build_ingested_file_stats(collection)
    ingested_files = set(ingested_stats.keys())

    print("\n=== File Ingestion Status ===")
    if not source_files:
        print("No source files found in DOCUMENTS_DIR.")
    else:
        for file_name in source_files:
            chunk_count = ingested_stats.get(file_name, 0)
            status = "INGESTED" if chunk_count > 0 else "PENDING/NOT_INGESTED"
            print(f"- {file_name} | status: {status} | chunks: {chunk_count}")

    # Include unknown-file chunks if metadata did not carry filenames.
    unknown_chunk_count = ingested_stats.get("unknown_file", 0)
    if unknown_chunk_count > 0:
        print(f"- unknown_file | status: INGESTED | chunks: {unknown_chunk_count}")

    print(f"\nSummary: discovered_files={len(source_files)}, ingested_files={len(ingested_files)}")

    # -----------------------------
    # Parse latest log for complete error details
    # -----------------------------
    latest_log_file = _find_latest_log_file(project_root)
    failures = _parse_failure_blocks_from_log(latest_log_file)

    # Export machine-readable status report for auditing and sharing.
    report_csv = _write_status_csv(logs_dir, source_files, ingested_stats, failures)

    print("\n=== Error Details (from latest log file) ===")
    if latest_log_file is None:
        print("No log file found in logs folder.")
    else:
        print(f"Latest log file: {latest_log_file}")

        if not failures:
            print("No file-level ingestion failures found in the latest log.")
        else:
            for file_name, blocks in failures.items():
                print("\n----------------------------------------")
                print(f"File: {file_name}")
                for attempt_no, block in enumerate(blocks, start=1):
                    print(f"Attempt #{attempt_no} full error:")
                    print(block)

    print("\n=== Report Export ===")
    print(f"CSV status report: {report_csv}")

    # -----------------------------
    # Show sample records for data inspection
    # -----------------------------
    if total_records == 0:
        print("\nCollection exists but has no records.")
        return

    sample_size = min(5, total_records)
    sample_data = collection.get(include=["documents", "metadatas"], limit=sample_size)

    print(f"\n=== Sample Records ({sample_size}) ===")
    ids = sample_data.get("ids", [])
    docs = sample_data.get("documents", [])
    metas = sample_data.get("metadatas", [])

    for i in range(sample_size):
        print("\n-------------------------------")
        print(f"Record {i + 1}")
        print(f"ID: {ids[i] if i < len(ids) else 'N/A'}")

        metadata = metas[i] if i < len(metas) else None
        print(f"Metadata: {metadata}")

        doc_text = docs[i] if i < len(docs) else ""
        preview = (doc_text or "")[:400].replace("\n", " ")
        print(f"Document preview: {preview}")


if __name__ == "__main__":
    main()
