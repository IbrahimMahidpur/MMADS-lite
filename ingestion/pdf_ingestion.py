"""
PDF Ingestion — uses PyMuPDF for text extraction.
Falls back to LLaVA (vision model via Ollama) for scanned/image PDFs.
"""
import logging
import time
from pathlib import Path

from multimodal_ds.config import OLLAMA_BASE_URL, VISION_MODEL
from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str) -> UnifiedDocument:
    """
    Extract text and structure from a PDF file.
    Strategy:
      1. Try PyMuPDF text extraction (fast, works for digital PDFs)
      2. If text is sparse (scanned PDF), use LLaVA vision model per page
    """
    import fitz  # PyMuPDF

    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.PDF,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="pdf_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        pdf = fitz.open(file_path)
        doc.page_count = len(pdf)
        all_text = []
        image_pages = []

        for page_num, page in enumerate(pdf):
            text = page.get_text().strip()
            if len(text) > 50:
                all_text.append(f"[Page {page_num + 1}]\n{text}")
            else:
                # Sparse text — flag for vision processing
                image_pages.append(page_num)

        doc.text_content = "\n\n".join(all_text)
        doc.metadata["total_pages"] = doc.page_count
        doc.metadata["text_pages"] = doc.page_count - len(image_pages)
        doc.metadata["image_pages"] = len(image_pages)

        # If >50% pages are image-based, use vision model
        if len(image_pages) > doc.page_count * 0.5:
            logger.info(f"[PDF] Scanned PDF detected — using vision model for {len(image_pages)} pages")
            vision_texts = _extract_with_vision(pdf, image_pages, file_path)
            doc.text_content += "\n\n" + "\n\n".join(vision_texts)
            doc.provenance.model_used = VISION_MODEL
        else:
            doc.provenance.model_used = "pymupdf"

        pdf.close()
        doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[PDF] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    logger.info(f"[PDF] Ingested {path.name} — {doc.page_count} pages, {len(doc.text_content)} chars in {doc.provenance.processing_time_s}s")
    return doc


def _extract_with_vision(pdf, page_nums: list[int], file_path: str) -> list[str]:
    """Use LLaVA via Ollama to describe image-based PDF pages."""
    import base64
    import httpx

    results = []
    for page_num in page_nums[:5]:  # Limit to first 5 image pages to avoid timeout
        try:
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": VISION_MODEL.replace("ollama/", ""),
                    "prompt": "Extract and describe all text, tables, charts, and figures visible in this document page. Be thorough and structured.",
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json().get("response", "")
                results.append(f"[Page {page_num + 1} — Vision]\n{text}")
        except Exception as e:
            logger.warning(f"[PDF Vision] Page {page_num} failed: {e}")

    return results
