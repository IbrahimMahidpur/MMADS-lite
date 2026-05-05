"""
Visualization Agent — generates rich Plotly charts with LLM narrative.
Specialist agent #4: converts analysis outputs into an insight gallery.

Responsibilities:
  - Detect what data/results exist in the session directory
  - Generate appropriate chart types per data shape
  - Write LLM-powered statistical narrative per chart
  - Publish VIZ_COMPLETE with chart manifest on the message bus
  - Callable standalone or auto-triggered by orchestrator post code-execution

Chart types supported:
  - Correlation heatmap (numeric features, requires >= 2 columns)
  - Distribution plots (histogram + KDE per numeric column)
  - Churn/target rate analysis (binary classification)
  - Feature importance bar chart (if model output exists)
  - ROC curve (if model probabilities exist)
  - Scatter matrix (numeric pairs, colored by target)
  - Time series decomposition (if datetime index detected)
  - Missing value heatmap (data quality)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd

from multimodal_ds.config import (
    OLLAMA_BASE_URL,
    REVIEWER_MODEL,
    LLM_TIMEOUT,
    OUTPUT_DIR,
)
from multimodal_ds.core.message_bus import (
    AgentMessage,
    MessageType,
    Priority,
    get_bus,
)
from multimodal_ds.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

# ── Plotly imports with graceful fallback ──────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio

    pio.templates.default = "plotly_dark"
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("[VizAgent] plotly not installed — charts will be skipped")


# ══════════════════════════════════════════════════════════════════════════
#  Chart Manifest
# ══════════════════════════════════════════════════════════════════════════

class ChartManifest:
    """Ordered list of charts + narratives for a session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.charts: list[dict[str, Any]] = []

    def add(
        self,
        chart_type: str,
        filename: str,
        title: str,
        narrative: str,
        data_shape: Optional[tuple] = None,
    ) -> None:
        self.charts.append(
            {
                "chart_type": chart_type,
                "filename": filename,
                "title": title,
                "narrative": narrative,
                "data_shape": list(data_shape) if data_shape else [],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "chart_count": len(self.charts),
            "charts": self.charts,
        }

    def save(self, output_dir: Path) -> Path:
        path = output_dir / "chart_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ══════════════════════════════════════════════════════════════════════════
#  Visualization Agent
# ══════════════════════════════════════════════════════════════════════════

class VisualizationAgent:
    """
    Generates a full insight gallery from session analysis outputs.

    Usage (standalone):
        agent = VisualizationAgent(session_id="abc123")
        manifest = agent.generate(df=df, target_col="churn")

    Usage (from orchestrator — via message bus):
        The orchestrator calls agent.generate() and the agent
        publishes VIZ_COMPLETE on completion.
    """

    AGENT_NAME = "visualization_agent"

    def __init__(
        self,
        session_id: str = "default",
        working_dir: Optional[str] = None,
    ):
        self.session_id = session_id
        self.working_dir = Path(working_dir or OUTPUT_DIR) / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.memory = AgentMemory()
        self.bus = get_bus()

    # ── Main Entry Point ───────────────────────────────────────────────────

    def generate(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        task_results: Optional[list[dict]] = None,
        stat_report: Optional[dict] = None,
    ) -> ChartManifest:
        """
        Generate all applicable charts for this dataset.

        Args:
            df:           Primary DataFrame (from tabular ingestion)
            target_col:   Binary/classification target column name
            task_results: Output from code execution tasks (for feature importance, ROC)
            stat_report:  Output from statistical agent (for normality callouts)

        Returns:
            ChartManifest with all generated chart metadata + narratives
        """
        if not _PLOTLY_AVAILABLE:
            logger.warning("[VizAgent] Plotly unavailable — returning empty manifest")
            return ChartManifest(self.session_id)

        # Guard: empty dataframe
        if df is None or df.empty:
            logger.warning("[VizAgent] Empty dataframe — returning empty manifest")
            return ChartManifest(self.session_id)

        t_start = time.time()
        manifest = ChartManifest(self.session_id)
        task_results = task_results or []

        logger.info(
            f"[VizAgent] Generating charts for session {self.session_id} "
            f"— shape {df.shape}, target={target_col}"
        )

        # Publish start event
        self.bus.publish(AgentMessage(
            msg_type=MessageType.VIZ_REQUEST,
            payload={
                "session_id": self.session_id,
                "df_shape": list(df.shape),
                "target_col": target_col,
            },
            sender=self.AGENT_NAME,
            session_id=self.session_id,
        ))

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        has_datetime = len(df.select_dtypes(include=["datetime64"]).columns) > 0

        # ── 1. Data Quality Heatmap ──────────────────────────────────────
        self._chart_missing_values(df, manifest)

        # ── 2. Distribution plots ────────────────────────────────────────
        if numeric_cols:
            self._chart_distributions(df, numeric_cols, target_col, manifest)

        # ── 3. Correlation heatmap (requires >= 2 numeric columns) ───────
        if len(numeric_cols) >= 2:
            self._chart_correlation_heatmap(df, numeric_cols, manifest)

        # ── 4. Target analysis (classification) ──────────────────────────
        if target_col and target_col in df.columns:
            self._chart_target_analysis(df, target_col, numeric_cols, manifest)

        # ── 5. Scatter matrix ─────────────────────────────────────────────
        if 2 <= len(numeric_cols) <= 8:
            self._chart_scatter_matrix(df, numeric_cols, target_col, manifest)

        # ── 6. Feature importance (from code agent pkl / CSV output) ─────
        self._chart_feature_importance(task_results, manifest)

        # ── 7. ROC curve ─────────────────────────────────────────────────
        self._chart_roc_curve(df, target_col, task_results, manifest)

        # ── 8. Time series decomposition ─────────────────────────────────
        if has_datetime and target_col:
            self._chart_time_series(df, target_col, manifest)

        # ── Save manifest ────────────────────────────────────────────────
        manifest_path = manifest.save(self.working_dir)

        # Store to memory
        self.memory.store_analysis_step(
            step_name="visualization",
            result=(
                f"Generated {len(manifest.charts)} charts: "
                + ", ".join(c["chart_type"] for c in manifest.charts)
            ),
            session_id=self.session_id,
        )

        duration = round(time.time() - t_start, 2)
        logger.info(
            f"[VizAgent] Session {self.session_id} — "
            f"{len(manifest.charts)} charts in {duration}s"
        )

        # Publish completion
        self.bus.publish(AgentMessage(
            msg_type=MessageType.VIZ_COMPLETE,
            payload={
                "chart_count": len(manifest.charts),
                "chart_types": [c["chart_type"] for c in manifest.charts],
                "manifest_path": str(manifest_path),
                "duration_s": duration,
            },
            sender=self.AGENT_NAME,
            session_id=self.session_id,
            priority=Priority.NORMAL,
        ))

        return manifest

    # ── Chart Generators ───────────────────────────────────────────────────

    def _chart_missing_values(self, df: pd.DataFrame, manifest: ChartManifest) -> None:
        """Missing value heatmap — data quality overview."""
        try:
            missing_pct = (df.isnull().mean() * 100).round(2)
            cols_with_missing = missing_pct[missing_pct > 0]

            if cols_with_missing.empty:
                fig = go.Figure(go.Bar(
                    x=list(missing_pct.index),
                    y=[100.0] * len(missing_pct),
                    marker_color="#00CC96",
                    text=["100% complete"] * len(missing_pct),
                    textposition="outside",
                ))
                title = "Data Completeness — All Columns 100% Complete"
                narrative_hint = "Dataset has zero missing values across all columns."
            else:
                fig = go.Figure(go.Bar(
                    x=list(missing_pct.index),
                    y=list(missing_pct.values),
                    marker_color=[
                        "#EF553B" if v > 20 else "#FFA15A" if v > 5 else "#636EFA"
                        for v in missing_pct.values
                    ],
                    text=[f"{v:.1f}%" for v in missing_pct.values],
                    textposition="outside",
                ))
                title = "Missing Value Analysis by Column"
                narrative_hint = (
                    f"{len(cols_with_missing)} columns have missing data. "
                    f"Worst: {cols_with_missing.idxmax()} at {cols_with_missing.max():.1f}%."
                )

            fig.update_layout(
                title=title,
                xaxis_title="Column",
                yaxis_title="Missing (%)",
                yaxis_range=[0, 110],
                height=400,
                template="plotly_dark",
            )

            filename = "data_quality_missing.html"
            fig.write_html(str(self.working_dir / filename))

            narrative = self._generate_narrative(
                chart_type="data_quality",
                data_summary=narrative_hint,
                stats={
                    "total_columns": len(df.columns),
                    "columns_with_missing": int(len(cols_with_missing)),
                    "max_missing_pct": float(cols_with_missing.max()) if not cols_with_missing.empty else 0.0,
                },
            )
            manifest.add("data_quality", filename, title, narrative, df.shape)

        except Exception as e:
            logger.warning(f"[VizAgent] Missing value chart failed: {e}")

    def _chart_distributions(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        target_col: Optional[str],
        manifest: ChartManifest,
    ) -> None:
        """Per-column distribution: histogram + KDE, split by target if available."""
        try:
            cols_to_plot = [c for c in numeric_cols if c != target_col][:6]
            n = len(cols_to_plot)
            if n == 0:
                return

            rows = max(1, (n + 1) // 2)
            fig = make_subplots(
                rows=rows,
                cols=2,
                subplot_titles=cols_to_plot,
                vertical_spacing=0.12,
            )

            colors = px.colors.qualitative.Plotly

            for idx, col in enumerate(cols_to_plot):
                row, col_pos = divmod(idx, 2)
                row += 1
                col_pos += 1

                data = df[col].dropna()

                if target_col and target_col in df.columns and df[target_col].nunique() <= 10:
                    for i, val in enumerate(sorted(df[target_col].unique())):
                        subset = df[df[target_col] == val][col].dropna()
                        fig.add_trace(
                            go.Histogram(
                                x=subset,
                                name=f"{target_col}={val}",
                                opacity=0.7,
                                marker_color=colors[i % len(colors)],
                                showlegend=(idx == 0),
                                legendgroup=str(val),
                            ),
                            row=row,
                            col=col_pos,
                        )
                else:
                    fig.add_trace(
                        go.Histogram(
                            x=data,
                            name=col,
                            marker_color=colors[idx % len(colors)],
                            showlegend=False,
                        ),
                        row=row,
                        col=col_pos,
                    )

            fig.update_layout(
                title_text="Feature Distributions",
                barmode="overlay",
                height=300 * rows,
                template="plotly_dark",
            )

            filename = "distributions.html"
            fig.write_html(str(self.working_dir / filename))

            skew_info = {
                col: round(float(df[col].skew()), 3)
                for col in cols_to_plot
                if col in df.columns
            }
            narrative = self._generate_narrative(
                chart_type="distributions",
                data_summary=f"Feature distributions for {len(cols_to_plot)} numeric columns.",
                stats={"skewness": skew_info, "split_by_target": target_col is not None},
            )
            manifest.add("distributions", filename, "Feature Distributions", narrative, df.shape)

        except Exception as e:
            logger.warning(f"[VizAgent] Distribution chart failed: {e}")

    def _chart_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        manifest: ChartManifest,
    ) -> None:
        """
        Pearson correlation heatmap.
        Requires at least 2 numeric columns — skips silently otherwise.
        """
        try:
            # ── GUARD: need >= 2 columns to compute a meaningful correlation ──
            valid_cols = [c for c in numeric_cols if c in df.columns]
            if len(valid_cols) < 2:
                logger.debug(
                    f"[VizAgent] Correlation heatmap skipped — "
                    f"only {len(valid_cols)} numeric column(s) available"
                )
                return

            corr_df = df[valid_cols].corr()

            # Additional guard: corr() can produce a 1x1 matrix if one col has
            # zero variance or all-NaN after dropna. Check actual matrix size.
            if corr_df.shape[0] < 2 or corr_df.shape[1] < 2:
                logger.debug("[VizAgent] Correlation heatmap skipped — matrix too small after corr()")
                return

            # Mask upper triangle for cleaner look
            mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
            masked = corr_df.where(~mask)

            fig = go.Figure(
                go.Heatmap(
                    z=masked.values,
                    x=list(corr_df.columns),
                    y=list(corr_df.index),
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=masked.round(2).values.astype(str),
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    hoverongaps=False,
                    colorbar=dict(title="Pearson r"),
                )
            )
            fig.update_layout(
                title="Pearson Correlation Heatmap",
                height=max(400, 80 * len(valid_cols)),
                template="plotly_dark",
            )

            filename = "correlation_heatmap.html"
            fig.write_html(str(self.working_dir / filename))

            # Find strongest correlations for narrative
            pairs = []
            for i in range(len(corr_df.columns)):
                for j in range(i + 1, len(corr_df.columns)):
                    pairs.append(
                        (
                            corr_df.columns[i],
                            corr_df.columns[j],
                            round(float(corr_df.iloc[i, j]), 3),
                        )
                    )
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs = pairs[:3]

            narrative = self._generate_narrative(
                chart_type="correlation_heatmap",
                data_summary=f"Correlation structure of {len(valid_cols)} numeric features.",
                stats={
                    "top_correlations": [
                        {"col1": p[0], "col2": p[1], "r": p[2]} for p in top_pairs
                    ],
                    "n_strong": sum(1 for p in pairs if abs(p[2]) > 0.7),
                },
            )
            manifest.add(
                "correlation_heatmap",
                filename,
                "Pearson Correlation Heatmap",
                narrative,
                corr_df.shape,
            )

        except Exception as e:
            logger.warning(f"[VizAgent] Correlation heatmap failed: {e}")

    def _chart_target_analysis(
        self,
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: list[str],
        manifest: ChartManifest,
    ) -> None:
        """
        For binary classification: class balance, feature vs target box plots.
        """
        try:
            feature_cols = [c for c in numeric_cols if c != target_col][:4]
            n_panels = 1 + len(feature_cols)

            fig = make_subplots(
                rows=1,
                cols=n_panels,
                subplot_titles=["Class Balance"] + [f"{c} by {target_col}" for c in feature_cols],
                horizontal_spacing=0.08,
            )

            value_counts = df[target_col].value_counts()
            fig.add_trace(
                go.Bar(
                    x=[str(v) for v in value_counts.index],
                    y=list(value_counts.values),
                    marker_color=["#636EFA", "#EF553B"],
                    text=[f"{v / len(df) * 100:.1f}%" for v in value_counts.values],
                    textposition="outside",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            colors = ["#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
            for i, feat in enumerate(feature_cols):
                for j, val in enumerate(sorted(df[target_col].unique())):
                    subset = df[df[target_col] == val][feat].dropna()
                    fig.add_trace(
                        go.Box(
                            y=list(subset),
                            name=f"{target_col}={val}",
                            marker_color=colors[j % len(colors)],
                            showlegend=(i == 0),
                            legendgroup=str(val),
                        ),
                        row=1,
                        col=i + 2,
                    )

            fig.update_layout(
                title_text=f"Target Analysis — {target_col}",
                height=500,
                template="plotly_dark",
                boxmode="group",
            )

            filename = "target_analysis.html"
            fig.write_html(str(self.working_dir / filename))

            class_balance = value_counts.to_dict()
            means_by_class = {}
            for feat in feature_cols:
                means_by_class[feat] = {
                    str(k): round(float(v), 2)
                    for k, v in df.groupby(target_col)[feat].mean().items()
                }

            narrative = self._generate_narrative(
                chart_type="target_analysis",
                data_summary=(
                    f"Class balance and feature distributions split by {target_col}. "
                    f"Classes: {dict(class_balance)}"
                ),
                stats={"class_balance": class_balance, "feature_means_by_class": means_by_class},
            )
            manifest.add(
                "target_analysis",
                filename,
                f"Target Analysis — {target_col}",
                narrative,
                df.shape,
            )

        except Exception as e:
            logger.warning(f"[VizAgent] Target analysis chart failed: {e}")

    def _chart_scatter_matrix(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        target_col: Optional[str],
        manifest: ChartManifest,
    ) -> None:
        """Scatter matrix (SPLOM) colored by target."""
        try:
            cols = [c for c in numeric_cols if c != target_col][:5]
            if len(cols) < 2:
                return

            plot_df = df[cols + ([target_col] if target_col else [])].dropna()
            if len(plot_df) < 2:
                return

            if target_col and target_col in plot_df.columns:
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=cols,
                    color=target_col,
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title="Scatter Matrix (colored by target)",
                    opacity=0.6,
                )
            else:
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=cols,
                    template="plotly_dark",
                    title="Scatter Matrix",
                    opacity=0.6,
                )

            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.update_layout(height=600)

            filename = "scatter_matrix.html"
            fig.write_html(str(self.working_dir / filename))

            narrative = self._generate_narrative(
                chart_type="scatter_matrix",
                data_summary=f"Pairwise relationships between {len(cols)} features.",
                stats={
                    "features": cols,
                    "colored_by": target_col,
                    "n_points": len(plot_df),
                },
            )
            manifest.add(
                "scatter_matrix",
                filename,
                "Scatter Matrix",
                narrative,
                plot_df.shape,
            )

        except Exception as e:
            logger.warning(f"[VizAgent] Scatter matrix failed: {e}")

    def _chart_feature_importance(
        self,
        task_results: list[dict],
        manifest: ChartManifest,
    ) -> None:
        """
        Look for feature importance data in session dir (CSVs/JSON from code agent).
        Renders a sorted horizontal bar chart.
        """
        try:
            importance_data = self._find_feature_importance()
            if not importance_data:
                return

            features = list(importance_data.keys())
            importances = list(importance_data.values())

            sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1])
            features_sorted = [p[0] for p in sorted_pairs]
            imp_sorted = [p[1] for p in sorted_pairs]

            fig = go.Figure(
                go.Bar(
                    x=imp_sorted,
                    y=features_sorted,
                    orientation="h",
                    marker=dict(
                        color=imp_sorted,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Importance"),
                    ),
                    text=[f"{v:.4f}" for v in imp_sorted],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                height=max(400, 40 * len(features)),
                template="plotly_dark",
                margin=dict(l=150),
            )

            filename = "feature_importance.html"
            fig.write_html(str(self.working_dir / filename))

            top_feature = features_sorted[-1] if features_sorted else "unknown"
            narrative = self._generate_narrative(
                chart_type="feature_importance",
                data_summary=f"Feature importance from trained model. Top feature: {top_feature}",
                stats={
                    "top_features": dict(sorted(importance_data.items(), key=lambda x: -x[1])[:5]),
                    "n_features": len(features),
                },
            )
            manifest.add(
                "feature_importance",
                filename,
                "Feature Importance",
                narrative,
            )

        except Exception as e:
            logger.warning(f"[VizAgent] Feature importance chart failed: {e}")

    def _chart_roc_curve(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        task_results: list[dict],
        manifest: ChartManifest,
    ) -> None:
        """
        ROC curve — fits a quick Logistic Regression baseline.
        Requires sklearn and at least 10 rows with 2 target classes.
        """
        if not target_col or target_col not in df.columns:
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col]
            if not feature_cols:
                return

            clean_df = df[feature_cols + [target_col]].dropna()
            if len(clean_df) < 10 or clean_df[target_col].nunique() < 2:
                return

            X = clean_df[feature_cols].values
            y = clean_df[target_col].values

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_train_s, y_train)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Random Classifier",
            ))
            fig.add_trace(go.Scatter(
                x=list(fpr),
                y=list(tpr),
                mode="lines",
                line=dict(color="#00CC96", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 204, 150, 0.15)",
                name=f"Logistic Regression (AUC = {roc_auc:.3f})",
            ))
            fig.update_layout(
                title=f"ROC Curve — {target_col} Prediction",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=500,
                template="plotly_dark",
            )

            filename = "roc_curve.html"
            fig.write_html(str(self.working_dir / filename))

            narrative = self._generate_narrative(
                chart_type="roc_curve",
                data_summary=(
                    f"Logistic Regression baseline ROC curve for {target_col} prediction. "
                    f"AUC = {roc_auc:.3f}"
                ),
                stats={
                    "auc": round(float(roc_auc), 4),
                    "model": "LogisticRegression (baseline)",
                    "test_size": len(y_test),
                    "n_features": len(feature_cols),
                },
            )
            manifest.add(
                "roc_curve",
                filename,
                f"ROC Curve — AUC {roc_auc:.3f}",
                narrative,
                clean_df.shape,
            )

        except ImportError:
            logger.warning("[VizAgent] sklearn not available — ROC curve skipped")
        except Exception as e:
            logger.warning(f"[VizAgent] ROC curve failed: {e}")

    def _chart_time_series(
        self,
        df: pd.DataFrame,
        target_col: str,
        manifest: ChartManifest,
    ) -> None:
        """Time series line chart if datetime index detected."""
        try:
            dt_df = df.copy()
            if not isinstance(dt_df.index, pd.DatetimeIndex):
                dt_cols = dt_df.select_dtypes(include=["datetime64"]).columns
                if dt_cols.empty:
                    return
                dt_df = dt_df.set_index(dt_cols[0])

            if target_col not in dt_df.columns:
                return

            fig = px.line(
                dt_df.reset_index(),
                x=dt_df.index.name or "index",
                y=target_col,
                title=f"Time Series — {target_col}",
                template="plotly_dark",
            )
            fig.update_layout(height=400)

            filename = "time_series.html"
            fig.write_html(str(self.working_dir / filename))

            narrative = self._generate_narrative(
                chart_type="time_series",
                data_summary=f"Time series of {target_col} over {len(dt_df)} time steps.",
                stats={
                    "n_points": len(dt_df),
                    "mean": round(float(dt_df[target_col].mean()), 3),
                    "trend": "increasing" if dt_df[target_col].iloc[-1] > dt_df[target_col].iloc[0] else "decreasing",
                },
            )
            manifest.add("time_series", filename, f"Time Series — {target_col}", narrative)

        except Exception as e:
            logger.warning(f"[VizAgent] Time series chart failed: {e}")

    # ── LLM Narrative ──────────────────────────────────────────────────────

    def _generate_narrative(
        self,
        chart_type: str,
        data_summary: str,
        stats: dict,
    ) -> str:
        """
        Generate a full-paragraph statistical narrative for a chart using Ollama.
        Falls back to a template string if LLM is unavailable.
        """
        stats_text = json.dumps(stats, indent=2)[:800]

        prompt = f"""You are a senior data scientist writing chart commentary for an executive audience.

Chart type: {chart_type}
Data summary: {data_summary}
Statistical details:
{stats_text}

Write a 2-3 sentence insight paragraph that:
1. States what the chart shows (the key finding)
2. Explains why it matters (business/modeling implication)
3. Recommends a concrete next step

Be specific — cite the actual numbers from the stats. No filler phrases."""

        model = REVIEWER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a concise data science writer. Write clear, specific chart narratives.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": 200, "temperature": 0.3},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.debug(f"[VizAgent] Narrative LLM call failed: {e}")

        # Fallback: template narrative
        return (
            f"{chart_type.replace('_', ' ').title()} — {data_summary} "
            f"Key statistics: {json.dumps(stats)[:200]}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find_feature_importance(self) -> dict[str, float]:
        """
        Scan session directory for feature importance artifacts.
        Looks for: feature_importance.csv, importance.json, or sklearn pkl.
        """
        import pickle

        # Check for CSV with feature importance columns
        for csv_path in self.working_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                feat_col = next(
                    (c for c in df.columns if "feature" in c.lower()), None
                )
                imp_col = next(
                    (c for c in df.columns if "importance" in c.lower() or "score" in c.lower()),
                    None,
                )
                if feat_col and imp_col:
                    return dict(zip(df[feat_col].astype(str), df[imp_col].astype(float)))
            except Exception:
                continue

        # Check for JSON importance files
        for json_path in self.working_dir.glob("*importance*.json"):
            try:
                data = json.loads(json_path.read_text())
                if isinstance(data, dict):
                    return {k: float(v) for k, v in data.items()}
            except Exception:
                continue

        # Try loading sklearn model pkl
        for pkl_path in self.working_dir.glob("*.pkl"):
            try:
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)
                if hasattr(model, "feature_importances_"):
                    companion_csv = list(self.working_dir.glob("*.csv"))
                    if companion_csv:
                        df = pd.read_csv(companion_csv[0])
                        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
                        importances = model.feature_importances_
                        if len(feature_cols) == len(importances):
                            return dict(zip(feature_cols, importances.tolist()))
                    return {
                        f"feature_{i}": float(v)
                        for i, v in enumerate(model.feature_importances_)
                    }
            except Exception:
                continue

        return {}
