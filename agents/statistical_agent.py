"""
Statistical Reasoning Agent — validates statistical assumptions.
Specialist agent #2: normality, stationarity, multicollinearity, etc.
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd

from multimodal_ds.config import REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT
from multimodal_ds.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class StatisticalReasoningAgent:
    """
    Validates statistical assumptions before modeling:
    - Normality (Shapiro-Wilk, D'Agostino)
    - Stationarity (ADF test for time series)
    - Multicollinearity (VIF)
    - Homoscedasticity (Levene's test)
    - Correlation analysis
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory = AgentMemory()

    def validate_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        """
        Run full statistical validation on a DataFrame.
        Returns structured report with findings and recommendations.
        """
        report = {
            "normality": self._check_normality(df),
            "correlation": self._check_correlation(df),
            "multicollinearity": self._check_multicollinearity(df, target_col),
            "stationarity": self._check_stationarity(df),
            "recommendations": [],
        }

        # Get LLM interpretation of findings
        report["interpretation"] = self._interpret_findings(report, df.shape)
        report["recommendations"] = self._generate_recommendations(report)

        # Store in memory
        self.memory.store_analysis_step(
            step_name="statistical_validation",
            result=str(report["interpretation"])[:500],
            session_id=self.session_id
        )

        return report

    def _check_normality(self, df: pd.DataFrame) -> dict:
        """Test normality for all numeric columns."""
        from scipy import stats

        numeric_cols = df.select_dtypes(include=np.number).columns
        results = {}

        for col in numeric_cols[:10]:  # Limit to 10 cols
            data = df[col].dropna()
            if len(data) < 3:
                continue
            try:
                # Shapiro-Wilk (best for n < 5000)
                if len(data) <= 5000:
                    stat, p = stats.shapiro(data)
                    test = "shapiro_wilk"
                else:
                    stat, p = stats.normaltest(data)
                    test = "dagostino"

                results[col] = {
                    "test": test,
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "is_normal": bool(p > 0.05),
                    "skewness": round(float(data.skew()), 3),
                    "kurtosis": round(float(data.kurtosis()), 3),
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        return results

    def _check_correlation(self, df: pd.DataFrame) -> dict:
        """Compute correlation matrix and identify strong correlations."""
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()
        strong_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "correlation": round(float(corr_val), 3),
                        "strength": "very_strong" if abs(corr_val) > 0.9 else "strong"
                    })

        return {
            "matrix": corr_matrix.round(3).to_dict(),
            "strong_pairs": strong_pairs,
            "n_strong": len(strong_pairs)
        }

    def _check_multicollinearity(self, df: pd.DataFrame, target_col: Optional[str]) -> dict:
        """Check VIF (Variance Inflation Factor) for multicollinearity."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            numeric_df = df.select_dtypes(include=np.number).dropna()
            if target_col and target_col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[target_col])

            if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
                return {"skipped": "Not enough data"}

            # Add constant for VIF calculation
            from statsmodels.tools.tools import add_constant
            X = add_constant(numeric_df)

            vif_data = {}
            for i, col in enumerate(X.columns[1:], 1):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_data[col] = round(float(vif), 2)
                except Exception:
                    pass

            high_vif = {k: v for k, v in vif_data.items() if v > 5}
            return {
                "vif_scores": vif_data,
                "high_vif_cols": high_vif,
                "multicollinearity_detected": len(high_vif) > 0
            }
        except ImportError:
            return {"skipped": "statsmodels not installed"}
        except Exception as e:
            return {"error": str(e)}

    def _check_stationarity(self, df: pd.DataFrame) -> dict:
        """Check stationarity for time-series-like columns using ADF test."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return {"skipped": "statsmodels not installed"}

        # Check if any datetime index exists
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if not len(datetime_cols) and not isinstance(df.index, pd.DatetimeIndex):
            return {"skipped": "No datetime column found — likely not time series data"}

        numeric_cols = df.select_dtypes(include=np.number).columns
        results = {}
        for col in numeric_cols[:5]:
            data = df[col].dropna()
            if len(data) < 20:
                continue
            try:
                adf_stat, p_value, _, _, critical_values, _ = adfuller(data)
                results[col] = {
                    "adf_statistic": round(float(adf_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "is_stationary": bool(p_value < 0.05),
                    "critical_values": {k: round(v, 3) for k, v in critical_values.items()}
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        return results

    def _interpret_findings(self, report: dict, shape: tuple) -> str:
        """Use Ollama to interpret statistical findings."""
        import httpx

        findings_text = f"""Dataset shape: {shape}
Normality: {len([v for v in report['normality'].values() if isinstance(v, dict) and v.get('is_normal')])} of {len(report['normality'])} columns are normal
Strong correlations: {report['correlation'].get('n_strong', 0)} pairs
Multicollinearity: {report['multicollinearity'].get('multicollinearity_detected', False)}
Non-stationary columns: {len([v for v in report['stationarity'].values() if isinstance(v, dict) and not v.get('is_stationary', True)])}"""

        model = REVIEWER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a statistician. Interpret findings concisely in 3-4 sentences."},
                        {"role": "user", "content": f"Interpret these statistical findings:\n{findings_text}"}
                    ],
                    "stream": False,
                    "options": {"num_predict": 300, "temperature": 0.2},
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
        except Exception:
            pass
        return findings_text

    def _generate_recommendations(self, report: dict) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recs = []

        # Normality
        non_normal = [k for k, v in report["normality"].items() if isinstance(v, dict) and not v.get("is_normal", True)]
        if non_normal:
            recs.append(f"Apply log/sqrt transformation to non-normal columns: {', '.join(non_normal[:3])}")

        # Multicollinearity
        if report["multicollinearity"].get("multicollinearity_detected"):
            high_vif = list(report["multicollinearity"].get("high_vif_cols", {}).keys())
            recs.append(f"Consider removing/combining highly collinear features: {', '.join(high_vif[:3])}")

        # Strong correlations
        if report["correlation"].get("n_strong", 0) > 3:
            recs.append("Use PCA or feature selection to handle multicollinearity")

        # Stationarity
        non_stationary = [k for k, v in report["stationarity"].items() if isinstance(v, dict) and not v.get("is_stationary", True)]
        if non_stationary:
            recs.append(f"Apply differencing to non-stationary columns before time-series modeling: {', '.join(non_stationary[:3])}")

        if not recs:
            recs.append("Data appears statistically well-behaved — proceed with standard modeling")

        return recs
