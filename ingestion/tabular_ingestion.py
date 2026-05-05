"""
Tabular Data Ingestion — pandas profiling + FLAML AutoML suggestions.
Handles CSV, Excel, Parquet, JSON tabular files.
"""
import logging
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)

SUPPORTED_TABULAR = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv"}


def ingest_tabular(file_path: str) -> UnifiedDocument:
    """
    Ingest tabular data with:
    - Schema detection
    - Statistical profiling
    - FLAML AutoML task suggestion
    - Missing value / outlier summary
    """
    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.TABULAR,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="tabular_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        df = _load_dataframe(file_path)
        if df is None:
            raise ValueError(f"Could not load dataframe from {file_path}")

        doc.structured_data = df

        # Schema info
        doc.schema_info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        }

        # Data profile
        doc.data_profile = _compute_profile(df)

        # Text summary for LLM consumption
        doc.text_content = _generate_text_summary(df, doc.schema_info, doc.data_profile)

        doc.metadata["automl_suggestion"] = _suggest_automl_task(df)
        doc.metadata["file_format"] = path.suffix.lower()

        doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[Tabular] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    logger.info(f"[Tabular] Ingested {path.name} in {doc.provenance.processing_time_s}s")
    return doc


def _load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load file into pandas DataFrame based on extension."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".csv":     lambda: pd.read_csv(file_path),
        ".tsv":     lambda: pd.read_csv(file_path, sep="\t"),
        ".xlsx":    lambda: pd.read_excel(file_path),
        ".xls":     lambda: pd.read_excel(file_path),
        ".parquet": lambda: pd.read_parquet(file_path),
        ".json":    lambda: pd.read_json(file_path),
    }
    loader = loaders.get(ext)
    return loader() if loader else None


def _compute_profile(df: pd.DataFrame) -> dict:
    """Compute statistical profile of the dataframe."""
    profile = {}
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        desc = numeric_df.describe()
        profile["numeric_stats"] = desc.to_dict()

    profile["missing_values"] = df.isnull().sum().to_dict()
    profile["missing_pct"] = (df.isnull().mean() * 100).round(2).to_dict()
    profile["duplicate_rows"] = int(df.duplicated().sum())
    profile["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2)

    # Cardinality for categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    profile["cardinality"] = {col: int(df[col].nunique()) for col in cat_cols}

    # Outlier detection (IQR method) for numeric cols
    outlier_counts = {}
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((numeric_df[col] < q1 - 1.5 * iqr) | (numeric_df[col] > q3 + 1.5 * iqr)).sum()
        if outliers > 0:
            outlier_counts[col] = int(outliers)
    profile["outlier_counts"] = outlier_counts

    return profile


def _generate_text_summary(df: pd.DataFrame, schema: dict, profile: dict) -> str:
    """Generate a natural language summary for LLM consumption."""
    rows, cols = schema["shape"]
    missing_total = sum(profile["missing_values"].values())
    top_missing = sorted(profile["missing_pct"].items(), key=lambda x: -x[1])[:3]

    lines = [
        f"Dataset: {rows} rows × {cols} columns",
        f"Numeric columns ({len(schema['numeric_cols'])}): {', '.join(schema['numeric_cols'][:5])}{'...' if len(schema['numeric_cols']) > 5 else ''}",
        f"Categorical columns ({len(schema['categorical_cols'])}): {', '.join(schema['categorical_cols'][:5])}",
        f"Missing values: {missing_total} total",
    ]
    if top_missing:
        lines.append("Columns with most missing: " + ", ".join(f"{c}={p:.1f}%" for c, p in top_missing))
    if profile.get("outlier_counts"):
        lines.append("Outliers detected in: " + ", ".join(profile["outlier_counts"].keys()))
    if profile.get("duplicate_rows"):
        lines.append(f"Duplicate rows: {profile['duplicate_rows']}")

    # Add describe stats
    if "numeric_stats" in profile:
        lines.append("\nNumeric Summary (mean ± std):")
        for col, stats in profile["numeric_stats"].items():
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            lines.append(f"  {col}: mean={mean:.3g}, std={std:.3g}, min={stats.get('min', 0):.3g}, max={stats.get('max', 0):.3g}")

    return "\n".join(lines)


def _suggest_automl_task(df: pd.DataFrame) -> dict:
    """Suggest ML task type based on data profile."""
    suggestion = {"task": "unknown", "target_candidates": [], "reason": ""}

    # Heuristic: last column often target
    last_col = df.columns[-1] if len(df.columns) > 0 else None
    if last_col:
        n_unique = df[last_col].nunique()
        if n_unique <= 20 and df[last_col].dtype in [object, "category"] or n_unique <= 10:
            suggestion["task"] = "classification"
            suggestion["target_candidates"] = [last_col]
            suggestion["reason"] = f"Last column '{last_col}' has {n_unique} unique values — likely classification target"
        elif pd.api.types.is_numeric_dtype(df[last_col]):
            suggestion["task"] = "regression"
            suggestion["target_candidates"] = [last_col]
            suggestion["reason"] = f"Last column '{last_col}' is numeric — likely regression target"

    return suggestion
