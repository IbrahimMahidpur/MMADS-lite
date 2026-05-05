"""
Unified schema for all ingested multimodal data.
Every modality (PDF, image, audio, tabular) outputs this schema.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class DataType(str, Enum):
    PDF       = "pdf"
    IMAGE     = "image"
    AUDIO     = "audio"
    TABULAR   = "tabular"
    TEXT      = "text"
    UNKNOWN   = "unknown"


class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class Provenance:
    """Track where data came from and how it was processed."""
    source_path: str
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processor: str = ""
    model_used: str = ""
    processing_time_s: float = 0.0
    raw_size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class UnifiedDocument:
    """
    Single output schema for ALL modalities.
    Every ingestion agent produces one of these.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_type: DataType = DataType.UNKNOWN
    status: ProcessingStatus = ProcessingStatus.PENDING

    # Core extracted content
    text_content: str = ""          # Extracted / transcribed text
    structured_data: Optional[Any] = None  # DataFrame for tabular, dict for others
    embeddings: Optional[list[float]] = None  # CLIP / nomic embeddings
    metadata: dict[str, Any] = field(default_factory=dict)

    # Provenance
    provenance: Provenance = field(default_factory=lambda: Provenance(source_path=""))

    # For tabular data
    schema_info: dict[str, Any] = field(default_factory=dict)  # dtypes, shape, nulls
    data_profile: dict[str, Any] = field(default_factory=dict)  # stats summary

    # For images/PDFs
    page_count: int = 0
    image_descriptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "data_type": self.data_type.value,
            "status": self.status.value,
            "text_content": self.text_content[:2000],  # Truncate for display
            "metadata": self.metadata,
            "schema_info": self.schema_info,
            "data_profile": self.data_profile,
            "page_count": self.page_count,
            "image_descriptions": self.image_descriptions,
            "provenance": {
                "source_path": self.provenance.source_path,
                "ingested_at": self.provenance.ingested_at,
                "processor": self.provenance.processor,
                "model_used": self.provenance.model_used,
            }
        }
