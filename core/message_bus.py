"""
Message Bus — typed inter-agent communication layer.

Every agent publishes and subscribes through this bus.
No agent calls another agent directly — all coordination
is mediated here, giving us:
  - Full audit trail of every agent interaction
  - Decoupled agents (easy to add/remove/replace)
  - Replay capability for debugging
  - Backpressure and priority queuing

Architecture:
    Agent → publish(AgentMessage) → MessageBus → subscriber callbacks
    Agent ← subscribe(MessageType) ← MessageBus ← other agents
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Message Types ──────────────────────────────────────────────────────────

class MessageType(str, Enum):
    # Lifecycle
    SESSION_START       = "session.start"
    SESSION_END         = "session.end"

    # Ingestion layer
    INGEST_REQUEST      = "ingest.request"
    INGEST_COMPLETE     = "ingest.complete"
    INGEST_FAILED       = "ingest.failed"

    # Planning layer
    PLAN_REQUEST        = "plan.request"
    PLAN_HYPOTHESIS     = "plan.hypothesis"
    PLAN_TASK_READY     = "plan.task_ready"
    PLAN_COMPLETE       = "plan.complete"

    # Statistical layer
    STATS_REQUEST       = "stats.request"
    STATS_COMPLETE      = "stats.complete"
    STATS_ANOMALY       = "stats.anomaly"          # High-priority: assumptions violated

    # Code execution layer
    CODE_REQUEST        = "code.request"
    CODE_COMPLETE       = "code.complete"
    CODE_FAILED         = "code.failed"
    CODE_RETRY          = "code.retry"

    # Visualization layer
    VIZ_REQUEST         = "viz.request"
    VIZ_COMPLETE        = "viz.complete"

    # Evaluation layer
    EVAL_REQUEST        = "eval.request"
    EVAL_COMPLETE       = "eval.complete"
    EVAL_FLAGGED        = "eval.flagged"           # High-priority: safety/quality issue

    # Cross-cutting
    AGENT_ERROR         = "agent.error"
    HANDOFF             = "agent.handoff"          # Explicit agent-to-agent handoff
    MEMORY_STORE        = "memory.store"
    MEMORY_RETRIEVE     = "memory.retrieve"


class Priority(int, Enum):
    LOW     = 0
    NORMAL  = 1
    HIGH    = 2
    URGENT  = 3    # Used for EVAL_FLAGGED, STATS_ANOMALY — preempts queue


# ── Core Message Dataclass ─────────────────────────────────────────────────

@dataclass
class AgentMessage:
    """
    Typed envelope for all inter-agent communication.

    Every message carries:
      - What kind of event this is (msg_type)
      - Who sent it and who should receive it
      - The actual payload
      - Correlation ID to trace a full analysis run
      - Causation ID to reconstruct the message chain
    """
    msg_type:       MessageType
    payload:        dict[str, Any]
    sender:         str                             # Agent name e.g. "planner", "code_agent"
    recipient:      Optional[str] = None            # None = broadcast to all subscribers
    session_id:     str = ""
    msg_id:         str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    causation_id:   Optional[str] = None            # msg_id of the message that triggered this
    priority:       Priority = Priority.NORMAL
    timestamp:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata:       dict[str, Any] = field(default_factory=dict)

    def reply(
        self,
        msg_type: MessageType,
        payload: dict[str, Any],
        sender: str,
        priority: Priority = Priority.NORMAL,
    ) -> "AgentMessage":
        """
        Construct a reply that preserves correlation chain.
        Use this instead of building a new AgentMessage manually.

            result_msg = incoming_msg.reply(
                MessageType.CODE_COMPLETE,
                payload={"files": [...]},
                sender="code_agent"
            )
        """
        return AgentMessage(
            msg_type=msg_type,
            payload=payload,
            sender=sender,
            recipient=self.sender,
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            causation_id=self.msg_id,
            priority=priority,
        )

    def to_dict(self) -> dict:
        return {
            "msg_id":         self.msg_id,
            "msg_type":       self.msg_type.value,
            "sender":         self.sender,
            "recipient":      self.recipient,
            "session_id":     self.session_id,
            "correlation_id": self.correlation_id,
            "causation_id":   self.causation_id,
            "priority":       self.priority.value,
            "timestamp":      self.timestamp,
            "payload_keys":   list(self.payload.keys()),  # Don't log full payload (PII risk)
            "metadata":       self.metadata,
        }


# ── Subscriber Type ────────────────────────────────────────────────────────

# A handler is any callable that accepts an AgentMessage
MessageHandler = Callable[[AgentMessage], None]


# ── Message Bus ───────────────────────────────────────────────────────────

class MessageBus:
    """
    Thread-safe, in-process message bus with:
      - Topic-based pub/sub (subscribe by MessageType)
      - Priority queue (URGENT messages jump the queue)
      - Full audit log per session
      - Dead-letter queue for failed deliveries
      - Middleware hooks (pre/post publish)

    Usage:
        bus = MessageBus()

        # Subscribe
        bus.subscribe(MessageType.CODE_COMPLETE, my_handler)

        # Publish
        bus.publish(AgentMessage(
            msg_type=MessageType.CODE_REQUEST,
            payload={"task": "..."},
            sender="orchestrator",
        ))

        # Inspect
        trace = bus.get_session_trace("abc123")
    """

    def __init__(self, max_audit_size: int = 10_000):
        self._subscribers: dict[MessageType, list[MessageHandler]] = defaultdict(list)
        self._wildcard_subscribers: list[MessageHandler] = []    # Subscribe to ALL types

        # Priority queue: deque per priority level (URGENT → LOW)
        self._queues: dict[Priority, deque[AgentMessage]] = {
            p: deque() for p in reversed(Priority)
        }

        # Audit log: session_id → list of message dicts (bounded)
        self._audit: dict[str, list[dict]] = defaultdict(list)
        self._max_audit_size = max_audit_size

        # Dead-letter queue: messages that had no subscribers or raised exceptions
        self._dlq: list[dict] = []

        # Middleware: list of callables run before dispatch
        self._middleware: list[Callable[[AgentMessage], Optional[AgentMessage]]] = []

        self._lock = threading.Lock()
        self._stats: dict[str, int] = defaultdict(int)

        logger.info("[MessageBus] Initialized")

    # ── Subscription API ───────────────────────────────────────────────────

    def subscribe(
        self,
        msg_type: MessageType,
        handler: MessageHandler,
        agent_name: str = "",
    ) -> None:
        """Register a handler for a specific message type."""
        with self._lock:
            self._subscribers[msg_type].append(handler)
        logger.debug(f"[Bus] {agent_name or handler.__name__} subscribed to {msg_type.value}")

    def subscribe_all(self, handler: MessageHandler, agent_name: str = "") -> None:
        """Register a handler that receives every message (useful for logging agents)."""
        with self._lock:
            self._wildcard_subscribers.append(handler)
        logger.debug(f"[Bus] {agent_name or handler.__name__} subscribed to ALL messages")

    def unsubscribe(self, msg_type: MessageType, handler: MessageHandler) -> None:
        # Use != not `is not` — bound methods (e.g. list.append) create a new
        # object on every attribute access, so identity comparison always fails.
        # __eq__ on bound methods correctly compares __self__ and __func__.
        with self._lock:
            self._subscribers[msg_type] = [
                h for h in self._subscribers[msg_type] if h != handler
            ]

    def add_middleware(
        self, fn: Callable[[AgentMessage], Optional[AgentMessage]]
    ) -> None:
        """
        Add a middleware function that runs before dispatch.
        Return None from middleware to DROP the message (e.g. PII filter).
        Return a (possibly modified) AgentMessage to continue.
        """
        self._middleware.append(fn)

    # ── Publish API ────────────────────────────────────────────────────────

    def publish(self, message: AgentMessage) -> bool:
        """
        Publish a message. Runs middleware, then dispatches synchronously
        to all registered handlers.

        Returns True if at least one handler received the message.
        """
        # Run middleware chain
        msg = message
        for mw in self._middleware:
            result = mw(msg)
            if result is None:
                logger.info(
                    f"[Bus] Message {msg.msg_id} ({msg.msg_type.value}) "
                    f"dropped by middleware"
                )
                self._stats["dropped"] += 1
                return False
            msg = result

        # Audit
        self._audit_message(msg)

        # Dispatch
        handlers = self._get_handlers(msg)

        if not handlers:
            self._dlq.append({
                "message": msg.to_dict(),
                "reason": "no_subscribers",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._stats["dlq"] += 1
            logger.warning(
                f"[Bus] No subscribers for {msg.msg_type.value} "
                f"from {msg.sender} — sent to DLQ"
            )
            return False

        delivered = 0
        for handler in handlers:
            try:
                handler(msg)
                delivered += 1
            except Exception as e:
                logger.error(
                    f"[Bus] Handler {handler.__name__} failed for "
                    f"{msg.msg_type.value}: {e}",
                    exc_info=True,
                )
                self._dlq.append({
                    "message": msg.to_dict(),
                    "handler": handler.__name__,
                    "error": str(e),
                    "reason": "handler_exception",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self._stats["handler_errors"] += 1

        self._stats["published"] += 1
        self._stats[f"type.{msg.msg_type.value}"] += 1
        return delivered > 0

    def publish_and_wait(
        self,
        message: AgentMessage,
        response_type: MessageType,
        timeout_s: float = 300.0,
    ) -> Optional[AgentMessage]:
        """
        Publish a message and block until a specific response type arrives
        on the same correlation_id. Useful for synchronous request/reply patterns.

            result = bus.publish_and_wait(
                request_msg,
                response_type=MessageType.CODE_COMPLETE,
                timeout_s=60,
            )
        """
        event = threading.Event()
        response_holder: list[AgentMessage] = []

        def waiter(msg: AgentMessage) -> None:
            if msg.correlation_id == message.correlation_id:
                response_holder.append(msg)
                event.set()

        self.subscribe(response_type, waiter)
        try:
            self.publish(message)
            fired = event.wait(timeout=timeout_s)
            if not fired:
                logger.warning(
                    f"[Bus] publish_and_wait timed out after {timeout_s}s "
                    f"waiting for {response_type.value}"
                )
                return None
            return response_holder[0] if response_holder else None
        finally:
            self.unsubscribe(response_type, waiter)

    # ── Introspection API ──────────────────────────────────────────────────

    def get_session_trace(self, session_id: str) -> list[dict]:
        """Return full ordered message trace for a session."""
        return list(self._audit.get(session_id, []))

    def get_stats(self) -> dict[str, int]:
        """Return publish/delivery statistics."""
        return dict(self._stats)

    def get_dlq(self) -> list[dict]:
        """Return dead-letter queue contents."""
        return list(self._dlq)

    def subscriber_count(self, msg_type: MessageType) -> int:
        return len(self._subscribers.get(msg_type, []))

    def clear_session(self, session_id: str) -> None:
        """Remove audit log for a session (call after session ends)."""
        self._audit.pop(session_id, None)

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_handlers(self, msg: AgentMessage) -> list[MessageHandler]:
        """
        Collect handlers in priority order:
          1. Exact recipient match (direct messages)
          2. Type subscribers
          3. Wildcard subscribers
        """
        handlers: list[MessageHandler] = []

        with self._lock:
            type_handlers = list(self._subscribers.get(msg.msg_type, []))
            wildcard = list(self._wildcard_subscribers)

        # If message has explicit recipient, filter to only that agent's handlers
        # (agent names are embedded via closure when subscribing — see orchestrator)
        handlers.extend(type_handlers)
        handlers.extend(wildcard)
        return handlers

    def _audit_message(self, msg: AgentMessage) -> None:
        if not msg.session_id:
            return
        audit_entry = msg.to_dict()
        session_log = self._audit[msg.session_id]
        if len(session_log) >= self._max_audit_size:
            session_log.pop(0)   # Evict oldest
        session_log.append(audit_entry)


# ── Handoff Protocol ───────────────────────────────────────────────────────

@dataclass
class HandoffContext:
    """
    Explicit handoff payload when one agent passes control to another.
    This is what FAANG interviewers mean when they ask
    "how do your agents know when to stop and pass to the next one?"
    """
    from_agent:     str
    to_agent:       str
    task:           dict[str, Any]
    data_context:   str
    prior_outputs:  list[dict[str, Any]] = field(default_factory=list)
    instructions:   str = ""
    constraints:    dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "from_agent":    self.from_agent,
            "to_agent":      self.to_agent,
            "task":          self.task,
            "data_context":  self.data_context,
            "prior_outputs": self.prior_outputs,
            "instructions":  self.instructions,
            "constraints":   self.constraints,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "HandoffContext":
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__ if k in payload})


# ── Singleton Bus ──────────────────────────────────────────────────────────
# One bus per process. Import this in every agent.

_bus_instance: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_bus() -> MessageBus:
    """
    Get the global MessageBus singleton.
    All agents import and use this — never instantiate MessageBus directly.

        from multimodal_ds.core.message_bus import get_bus
        bus = get_bus()
    """
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = MessageBus()
    return _bus_instance


def reset_bus() -> None:
    """Reset the singleton — used in tests only."""
    global _bus_instance
    with _bus_lock:
        _bus_instance = None
