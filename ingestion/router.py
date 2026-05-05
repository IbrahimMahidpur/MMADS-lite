"""
Ingestion Router Agent — detects file type and routes to correct ingestion module.
This is the entry point for ALL data ingestion in the system.
"""
import logging
from pathlib import Path
from typing import Union

from multimodal_ds.core.schema import DataType, UnifiedDocument
from multimodal_ds.ingestion.pdf_ingestion import ingest_pdf
from multimodal_ds.ingestion.audio_ingestion import ingest_audio, SUPPORTED_AUDIO
from multimodal_ds.ingestion.image_ingestion import ingest_image, SUPPORTED_IMAGES
from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular, SUPPORTED_TABULAR

logger = logging.getLogger(__name__)


def route_and_ingest(file_path: str) -> UnifiedDocument:
    """
    Main entry point. Detects file type and routes to appropriate ingestion pipeline.
    Returns a UnifiedDocument regardless of input type.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    logger.info(f"[Router] Ingesting {path.name} (type: {ext})")

    if ext == ".pdf":
        return ingest_pdf(file_path)

    elif ext in SUPPORTED_AUDIO:
        return ingest_audio(file_path)

    elif ext in SUPPORTED_IMAGES:
        return ingest_image(file_path)

    elif ext in SUPPORTED_TABULAR:
        return ingest_tabular(file_path)

    elif ext in {".txt", ".md", ".rst"}:
        return _ingest_plain_text(file_path)

    else:
        logger.warning(f"[Router] Unknown file type: {ext} — attempting text ingestion")
        return _ingest_plain_text(file_path)


def ingest_multiple(file_paths: list[str]) -> list[UnifiedDocument]:
    """Ingest multiple files and return list of UnifiedDocuments."""
    results = []
    for fp in file_paths:
        try:
            doc = route_and_ingest(fp)
            results.append(doc)
            logger.info(f"[Router] [OK] {Path(fp).name} -> {doc.data_type.value} ({doc.status.value})")
        except Exception as e:
            logger.error(f"[Router] [ERROR] Failed to ingest {fp}: {e}")
    return results


def _ingest_plain_text(file_path: str) -> UnifiedDocument:
    """Simple text file ingestion."""
    from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument
    import time

    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.TEXT,
        provenance=Provenance(
            source_path=str(path),
            processor="plain_text",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )
    try:
        doc.text_content = path.read_text(encoding="utf-8", errors="replace")
        doc.metadata["char_count"] = len(doc.text_content)
        doc.metadata["word_count"] = len(doc.text_content.split())
        doc.status = ProcessingStatus.DONE
    except Exception as e:
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)
    return doc
