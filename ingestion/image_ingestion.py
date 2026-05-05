"""
Image Ingestion — CLIP embeddings + LLaVA description via local Ollama.
No external APIs needed.
"""
import logging
import time
from pathlib import Path
from typing import Optional

from multimodal_ds.config import OLLAMA_BASE_URL, VISION_MODEL
from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def ingest_image(file_path: str) -> UnifiedDocument:
    """
    Process an image:
    1. Generate CLIP embeddings for semantic search
    2. Use LLaVA (via Ollama) for natural language description
    """
    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.IMAGE,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="image_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        from PIL import Image

        img = Image.open(file_path)
        doc.metadata["width"] = img.width
        doc.metadata["height"] = img.height
        doc.metadata["mode"] = img.mode
        doc.metadata["format"] = img.format

        # Step 1: CLIP embeddings
        embeddings = _get_clip_embeddings(img)
        if embeddings is not None:
            doc.embeddings = embeddings
            doc.provenance.model_used = "clip-vit-base-patch32"

        # Step 2: LLaVA description
        description = _describe_with_llava(file_path)
        doc.text_content = description
        doc.image_descriptions = [description]
        doc.metadata["llava_model"] = VISION_MODEL

        doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[Image] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    return doc


def _get_clip_embeddings(img) -> Optional[list[float]]:
    """Generate CLIP embeddings locally using transformers library."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].tolist()

    except Exception as e:
        logger.warning(f"[CLIP] Embedding failed (transformers not available?): {e}")
        return None


def _describe_with_llava(file_path: str) -> str:
    """Use LLaVA via Ollama to describe the image."""
    import base64
    import httpx

    try:
        with open(file_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        model_name = VISION_MODEL.replace("ollama/", "")
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": (
                    "Describe this image in detail. Include: "
                    "1) Main subject/content, "
                    "2) Any text visible, "
                    "3) Charts, graphs, or data if present, "
                    "4) Key visual features relevant for data analysis."
                ),
                "images": [img_b64],
                "stream": False,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"[LLaVA description unavailable: HTTP {response.status_code}]"

    except Exception as e:
        logger.warning(f"[LLaVA] Description failed: {e}")
        return f"[Image: {Path(file_path).name}] — Vision description unavailable"
