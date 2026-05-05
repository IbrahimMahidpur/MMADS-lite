"""
Agent Orchestrator — coordinates all agents via MessageBus.

Flow:
  1. Publish SESSION_START
  2. Ingest files → INGEST_COMPLETE per doc
  3. Statistical validation → STATS_COMPLETE (tabular only)
  4. Planner → PLAN_COMPLETE
  5. Per task: HANDOFF → CODE_COMPLETE (or CODE_FAILED)
  6. Publish SESSION_END with full result

All agent coordination is mediated through the MessageBus.
No agent calls another agent directly.
"""
import logging
import time
import uuid
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from multimodal_ds.core.message_bus import (
    AgentMessage, HandoffContext, MessageBus, MessageType, Priority, get_bus
)
from multimodal_ds.ingestion.router import ingest_multiple
from multimodal_ds.agents.planner_agent import run_planner
from multimodal_ds.agents.code_execution_agent import CodeExecutionAgent
from multimodal_ds.agents.statistical_agent import StatisticalReasoningAgent
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.schema import UnifiedDocument, DataType

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Complete result of an orchestration run."""
    session_id: str
    status: str
    objective: str
    documents: list[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    statistical_report: dict = field(default_factory=dict)
    task_results: list[dict] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    message_trace: list[dict] = field(default_factory=list)   # ← NEW: full bus trace
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id":        self.session_id,
            "status":            self.status,
            "objective":         self.objective,
            "documents_ingested": len(self.documents),
            "tasks_planned":     len(self.plan.get("analysis_plan", [])),
            "tasks_completed":   sum(1 for t in self.task_results if t.get("success")),
            "tasks_failed":      sum(1 for t in self.task_results if not t.get("success")),
            "files_created":     self.files_created,
            "statistical_report": self.statistical_report,
            "plan_summary":      self.plan.get("final_plan", ""),
            "task_results":      self.task_results,
            "errors":            self.errors,
            "message_trace":     self.message_trace,
            "duration_s":        round(self.duration_s, 2),
        }


class AgentOrchestrator:
    """
    Top-level coordinator. Wires agents together via MessageBus.

    Design principles:
    - Orchestrator owns the bus wiring (subscribe/publish)
    - Agents are stateless workers — they receive a message, do work, publish result
    - All state lives in RunResult + MessageBus audit log
    - Any agent can be swapped without touching others

    Usage:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(
            file_paths=["data/sales.csv"],
            objective="Predict churn and explain key drivers"
        )
    """

    AGENT_NAME = "orchestrator"

    def __init__(self, working_dir: Optional[str] = None):
        from multimodal_ds.config import OUTPUT_DIR
        self.working_dir = Path(working_dir or OUTPUT_DIR)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.memory = AgentMemory()
        self.bus: MessageBus = get_bus()
        self._wire_subscriptions()

    def _wire_subscriptions(self) -> None:
        """
        Register orchestrator as subscriber for key message types.
        The orchestrator listens to completion/failure events and drives
        the next step — it never polls; it reacts.
        """
        self.bus.subscribe(MessageType.AGENT_ERROR, self._on_agent_error, self.AGENT_NAME)
        self.bus.subscribe(MessageType.EVAL_FLAGGED, self._on_eval_flagged, self.AGENT_NAME)
        self.bus.subscribe(MessageType.STATS_ANOMALY, self._on_stats_anomaly, self.AGENT_NAME)
        logger.info("[Orchestrator] Subscribed to bus events")

    # ── Event Handlers ─────────────────────────────────────────────────────

    def _on_agent_error(self, msg: AgentMessage) -> None:
        """React to any agent reporting a hard error."""
        logger.error(
            f"[Orchestrator] Agent error from {msg.sender}: "
            f"{msg.payload.get('error', 'unknown')}"
        )

    def _on_eval_flagged(self, msg: AgentMessage) -> None:
        """React when evaluation agent flags a quality/safety issue."""
        logger.warning(
            f"[Orchestrator] Eval flagged issue: {msg.payload.get('reason', '')}"
        )

    def _on_stats_anomaly(self, msg: AgentMessage) -> None:
        """React when statistical agent detects violated assumptions."""
        logger.warning(
            f"[Orchestrator] Statistical anomaly: {msg.payload.get('anomaly', '')}"
        )

    # ── Main Run ───────────────────────────────────────────────────────────

    def run(
        self,
        file_paths: list[str],
        objective: str,
        session_id: Optional[str] = None,
        run_statistical_checks: bool = True,
        max_tasks: int = 6,
    ) -> RunResult:
        """
        Execute the full agentic pipeline end-to-end via message bus.

        Every major transition is a published message — giving us
        a complete, replayable audit trail of the entire run.
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        t_start = time.time()
        result = RunResult(session_id=session_id, status="running", objective=objective)

        logger.info(f"[Orchestrator] Session {session_id} — {objective}")

        # ── SESSION START ──────────────────────────────────────────────────
        self.bus.publish(AgentMessage(
            msg_type=MessageType.SESSION_START,
            payload={"objective": objective, "file_count": len(file_paths)},
            sender=self.AGENT_NAME,
            session_id=session_id,
            priority=Priority.HIGH,
        ))

        # ── STEP 1: INGEST ─────────────────────────────────────────────────
        documents = self._run_ingestion(file_paths, session_id, result)
        if not documents:
            return self._finalize(result, t_start, session_id)

        # ── STEP 2: STATISTICAL VALIDATION ────────────────────────────────
        if run_statistical_checks:
            self._run_statistical_validation(documents, session_id, result)

        # ── STEP 3: PLAN ───────────────────────────────────────────────────
        plan_ok = self._run_planning(documents, objective, session_id, result)
        if not plan_ok:
            return self._finalize(result, t_start, session_id)

        # ── STEP 4: EXECUTE TASKS ──────────────────────────────────────────
        self._run_task_execution(documents, session_id, result, max_tasks)

        return self._finalize(result, t_start, session_id)

    # ── Pipeline Steps ─────────────────────────────────────────────────────

    def _run_ingestion(
        self,
        file_paths: list[str],
        session_id: str,
        result: RunResult,
    ) -> list[UnifiedDocument]:
        """Ingest files and publish INGEST_COMPLETE per document."""
        logger.info(f"[Orchestrator] Ingesting {len(file_paths)} file(s)")

        self.bus.publish(AgentMessage(
            msg_type=MessageType.INGEST_REQUEST,
            payload={"file_paths": file_paths},
            sender=self.AGENT_NAME,
            session_id=session_id,
        ))

        try:
            documents = ingest_multiple(file_paths)
        except Exception as e:
            result.errors.append(f"Ingestion failed: {e}")
            self.bus.publish(AgentMessage(
                msg_type=MessageType.INGEST_FAILED,
                payload={"error": str(e)},
                sender=self.AGENT_NAME,
                session_id=session_id,
                priority=Priority.HIGH,
            ))
            result.status = "failed"
            return []

        result.documents = [doc.to_dict() for doc in documents]

        # Publish one INGEST_COMPLETE per document
        for doc in documents:
            self.bus.publish(AgentMessage(
                msg_type=MessageType.INGEST_COMPLETE,
                payload={
                    "document_id": doc.id,
                    "data_type": doc.data_type.value,
                    "status": doc.status.value,
                    "source": doc.provenance.source_path,
                    "schema_info": doc.schema_info,
                },
                sender=self.AGENT_NAME,
                session_id=session_id,
            ))

        logger.info(f"[Orchestrator] Ingested {len(documents)} document(s)")
        return documents

    def _run_statistical_validation(
        self,
        documents: list[UnifiedDocument],
        session_id: str,
        result: RunResult,
    ) -> None:
        """Run StatisticalReasoningAgent on tabular data, publish results."""
        tabular_docs = [
            d for d in documents
            if d.data_type == DataType.TABULAR and d.structured_data is not None
        ]

        if not tabular_docs:
            logger.info("[Orchestrator] No tabular data — skipping statistical validation")
            return

        logger.info(f"[Orchestrator] Statistical validation on {len(tabular_docs)} dataset(s)")

        self.bus.publish(AgentMessage(
            msg_type=MessageType.STATS_REQUEST,
            payload={"document_ids": [d.id for d in tabular_docs]},
            sender=self.AGENT_NAME,
            session_id=session_id,
        ))

        stat_agent = StatisticalReasoningAgent(session_id=session_id)
        try:
            primary_df = tabular_docs[0].structured_data
            stat_report = stat_agent.validate_dataset(primary_df)
            result.statistical_report = {
                k: v for k, v in stat_report.items()
                if k != "interpretation" or isinstance(v, str)
            }

            recs = stat_report.get("recommendations", [])

            self.bus.publish(AgentMessage(
                msg_type=MessageType.STATS_COMPLETE,
                payload={
                    "recommendations": recs,
                    "multicollinearity_detected": stat_report.get(
                        "multicollinearity", {}
                    ).get("multicollinearity_detected", False),
                    "non_normal_cols": [
                        k for k, v in stat_report.get("normality", {}).items()
                        if isinstance(v, dict) and not v.get("is_normal", True)
                    ],
                },
                sender="statistical_agent",
                session_id=session_id,
            ))

            # Check for anomalies worth surfacing as high-priority
            anomalies = self._detect_anomalies(stat_report)
            for anomaly in anomalies:
                self.bus.publish(AgentMessage(
                    msg_type=MessageType.STATS_ANOMALY,
                    payload={"anomaly": anomaly},
                    sender="statistical_agent",
                    session_id=session_id,
                    priority=Priority.HIGH,
                ))

        except Exception as e:
            logger.warning(f"[Orchestrator] Statistical check failed (non-fatal): {e}")
            result.errors.append(f"Statistical check warning: {e}")

    def _run_planning(
        self,
        documents: list[UnifiedDocument],
        objective: str,
        session_id: str,
        result: RunResult,
    ) -> bool:
        """Run planner and publish PLAN_COMPLETE."""
        logger.info("[Orchestrator] Generating analysis plan")

        self.bus.publish(AgentMessage(
            msg_type=MessageType.PLAN_REQUEST,
            payload={"objective": objective},
            sender=self.AGENT_NAME,
            session_id=session_id,
        ))

        try:
            plan_state = run_planner(
                user_objective=objective,
                documents=documents,
                session_id=session_id,
            )
            result.plan = {
                "hypotheses":    plan_state.get("hypotheses", []),
                "analysis_plan": plan_state.get("analysis_plan", []),
                "final_plan":    plan_state.get("final_plan", ""),
            }

            # Publish each hypothesis as its own message
            for hyp in result.plan["hypotheses"]:
                self.bus.publish(AgentMessage(
                    msg_type=MessageType.PLAN_HYPOTHESIS,
                    payload={"hypothesis": hyp},
                    sender="planner_agent",
                    session_id=session_id,
                ))

            self.bus.publish(AgentMessage(
                msg_type=MessageType.PLAN_COMPLETE,
                payload={
                    "task_count": len(result.plan["analysis_plan"]),
                    "hypothesis_count": len(result.plan["hypotheses"]),
                },
                sender="planner_agent",
                session_id=session_id,
            ))

            logger.info(f"[Orchestrator] Plan: {len(result.plan['analysis_plan'])} tasks")
            return True

        except Exception as e:
            result.errors.append(f"Planning failed: {e}")
            result.status = "partial"
            self.bus.publish(AgentMessage(
                msg_type=MessageType.AGENT_ERROR,
                payload={"error": str(e), "agent": "planner"},
                sender=self.AGENT_NAME,
                session_id=session_id,
                priority=Priority.HIGH,
            ))
            return False

    def _run_task_execution(
        self,
        documents: list[UnifiedDocument],
        session_id: str,
        result: RunResult,
        max_tasks: int,
    ) -> None:
        """Execute tasks via CodeExecutionAgent with HANDOFF protocol."""
        session_dir = self.working_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Copy source files into session dir so code agent can load them
        for doc in documents:
            src = Path(doc.provenance.source_path)
            if src.exists():
                shutil.copy2(src, session_dir / src.name)

        code_agent = CodeExecutionAgent(
            working_dir=str(session_dir),
            session_id=session_id,
        )
        data_context = self._build_data_context(documents)
        tasks = result.plan["analysis_plan"][:max_tasks]
        all_files: list[str] = []

        for task in tasks:
            task_name = task.get("name", f"step_{task.get('step', '?')}")

            # Build handoff context — explicit, typed, structured
            handoff = HandoffContext(
                from_agent=self.AGENT_NAME,
                to_agent="code_execution_agent",
                task=task,
                data_context=data_context,
                prior_outputs=[
                    {
                        "name": t["name"],
                        "output": t.get("output_preview", ""),
                        "files": t.get("files_created", []),
                    }
                    for t in result.task_results if t.get("success")
                ],
                instructions=(
                    "Save all outputs to the current directory. "
                    "Use simple filenames. Handle exceptions gracefully."
                ),
                constraints={"max_runtime_s": 300, "no_network": True},
            )

            # Publish HANDOFF message
            handoff_msg = AgentMessage(
                msg_type=MessageType.HANDOFF,
                payload=handoff.to_payload(),
                sender=self.AGENT_NAME,
                recipient="code_execution_agent",
                session_id=session_id,
            )
            self.bus.publish(handoff_msg)

            # Publish CODE_REQUEST
            self.bus.publish(AgentMessage(
                msg_type=MessageType.CODE_REQUEST,
                payload={"task": task, "task_name": task_name},
                sender=self.AGENT_NAME,
                session_id=session_id,
                correlation_id=handoff_msg.correlation_id,
            ))

            logger.info(f"[Orchestrator] Executing: {task_name}")
            try:
                task_result = code_agent.execute_task(task, data_context=data_context)
            except Exception as e:
                logger.error(f"[Orchestrator] Task '{task_name}' raised: {e}")
                result.errors.append(f"Task '{task_name}' system error: {e}")
                self.bus.publish(AgentMessage(
                    msg_type=MessageType.CODE_FAILED,
                    payload={"task_name": task_name, "error": str(e)},
                    sender="code_execution_agent",
                    session_id=session_id,
                    priority=Priority.HIGH,
                    correlation_id=handoff_msg.correlation_id,
                ))
                result.task_results.append({
                    "step": task.get("step"),
                    "name": task_name,
                    "success": False,
                    "output_preview": "",
                    "files_created": [],
                    "error": str(e),
                })
                continue

            # Publish result message
            completion_type = (
                MessageType.CODE_COMPLETE if task_result["success"]
                else MessageType.CODE_FAILED
            )
            self.bus.publish(AgentMessage(
                msg_type=completion_type,
                payload={
                    "task_name":     task_name,
                    "files_created": task_result["files_created"],
                    "output":        task_result["output"][:300],
                    "success":       task_result["success"],
                },
                sender="code_execution_agent",
                session_id=session_id,
                priority=Priority.NORMAL,
                correlation_id=handoff_msg.correlation_id,
            ))

            if not task_result["success"]:
                result.errors.append(
                    f"Task '{task_name}' failed: {task_result.get('error', 'Unknown')}"
                )

            task_record = {
                "step":           task.get("step"),
                "name":           task_name,
                "success":        task_result["success"],
                "output_preview": task_result["output"][:300],
                "files_created":  task_result["files_created"],
                "error":          task_result.get("error", ""),
            }
            result.task_results.append(task_record)
            all_files.extend(task_result["files_created"])

            # Enrich data_context for the next task
            if task_result["success"] and task_result["output"]:
                data_context += f"\n\n[{task_name} output]\n{task_result['output'][:500]}"

        result.files_created = list(dict.fromkeys(all_files))

    # ── Finalization ───────────────────────────────────────────────────────

    def _finalize(self, result: RunResult, t_start: float, session_id: str) -> RunResult:
        succeeded = sum(1 for t in result.task_results if t["success"])
        total = len(result.task_results)

        if result.status not in ("failed",):
            result.status = (
                "success" if succeeded == total and total > 0
                else "partial" if succeeded > 0
                else "failed"
            )

        result.duration_s = time.time() - t_start

        # Attach full message trace to result
        result.message_trace = self.bus.get_session_trace(session_id)

        self.bus.publish(AgentMessage(
            msg_type=MessageType.SESSION_END,
            payload={
                "status":     result.status,
                "tasks_ok":   succeeded,
                "tasks_total": total,
                "duration_s": result.duration_s,
            },
            sender=self.AGENT_NAME,
            session_id=session_id,
            priority=Priority.HIGH,
        ))

        # Persist summary to memory
        self.memory.store(
            content=(
                f"Run {session_id}: {result.objective}\n"
                f"Status: {result.status}\nTasks: {succeeded}/{total}"
            ),
            metadata={"type": "run_summary", "session_id": session_id},
            doc_id=f"run_{session_id}",
        )

        logger.info(
            f"[Orchestrator] Session {session_id} done — "
            f"{succeeded}/{total} tasks in {result.duration_s:.1f}s | "
            f"Bus stats: {self.bus.get_stats()}"
        )
        return result

    # ── Utilities ──────────────────────────────────────────────────────────

    def _build_data_context(self, documents: list[UnifiedDocument]) -> str:
        parts = ["IMPORTANT: All files listed below are in the CURRENT DIRECTORY."]
        for i, doc in enumerate(documents):
            filename = Path(doc.provenance.source_path).name
            header = f"[File {i+1}: {filename} (Type: {doc.data_type.value})]"
            body = doc.text_content[:1000] if doc.text_content else ""
            if doc.schema_info:
                body += f"\nSchema: {doc.schema_info}"
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)

    def _detect_anomalies(self, stat_report: dict) -> list[str]:
        """Surface statistical findings that warrant high-priority attention."""
        anomalies = []
        mc = stat_report.get("multicollinearity", {})
        if mc.get("multicollinearity_detected"):
            high_vif = mc.get("high_vif_cols", {})
            anomalies.append(
                f"Severe multicollinearity detected in: {list(high_vif.keys())}"
            )
        non_stationary = [
            k for k, v in stat_report.get("stationarity", {}).items()
            if isinstance(v, dict) and not v.get("is_stationary", True)
        ]
        if len(non_stationary) > 2:
            anomalies.append(
                f"Non-stationary columns detected: {non_stationary} — "
                "differencing required before time-series modeling"
            )
        return anomalies
