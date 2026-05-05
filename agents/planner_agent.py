"""
Hypothesis Generation + Planning Agent — LangGraph + ReAct reasoning.
Decomposes user objectives into analysis task sequences.
Uses Ollama for all LLM calls — no API keys needed.
"""
import json
import logging
from typing import Any, TypedDict, Annotated
import operator

from multimodal_ds.config import PLANNER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.schema import UnifiedDocument

logger = logging.getLogger(__name__)


class PlannerState(TypedDict):
    """LangGraph state for the planner agent."""
    session_id: str
    user_objective: str
    data_profiles: list[dict]          # From ingested documents
    analysis_plan: list[dict]          # Generated task sequence
    current_step: int
    messages: Annotated[list, operator.add]
    hypotheses: list[str]
    final_plan: str
    error: str


def _call_ollama(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call Ollama with a prompt and return response text."""
    import httpx
    model = PLANNER_MODEL.replace("ollama/", "")
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            },
            timeout=LLM_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        return f"[Error: HTTP {response.status_code}]"
    except Exception as e:
        return f"[Error: {e}]"


def generate_hypotheses(state: PlannerState) -> PlannerState:
    """Node: Generate initial hypotheses from data profiles."""
    profiles_text = json.dumps(state["data_profiles"], indent=2)[:3000]

    prompt = f"""You are an expert data scientist. Given this data profile and user objective, generate 3-5 specific, testable hypotheses.

User Objective: {state['user_objective']}

Data Profile:
{profiles_text}

Generate hypotheses as a JSON array. Each hypothesis should have:
- "id": short identifier
- "statement": the hypothesis
- "analysis_method": how to test it
- "expected_outcome": what success looks like

Respond ONLY with valid JSON array, no other text."""

    response = _call_ollama(prompt, system="You are a data science hypothesis generator. Output only valid JSON.")
    
    try:
        # Clean response to extract JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        hypotheses = json.loads(response)
        state["hypotheses"] = [h.get("statement", str(h)) for h in hypotheses]
        logger.info(f"[Planner] Generated {len(state['hypotheses'])} hypotheses")
    except Exception as e:
        logger.warning(f"[Planner] Hypothesis JSON parse failed: {e}")
        state["hypotheses"] = [response]

    return state


def decompose_into_tasks(state: PlannerState) -> PlannerState:
    """Node: Decompose objective into ordered analysis tasks."""
    hypotheses_text = "\n".join(f"- {h}" for h in state.get("hypotheses", []))
    profiles_text = json.dumps(state["data_profiles"], indent=2)[:2000]

    prompt = f"""You are a senior data scientist creating an analysis plan.

Objective: {state['user_objective']}

Hypotheses to test:
{hypotheses_text}

Data available:
{profiles_text}

Create a detailed analysis plan as a JSON array of tasks. Each task:
{{
  "step": 1,
  "name": "task name",
  "type": "eda|feature_engineering|modeling|evaluation|visualization|reporting",
  "description": "what to do",
  "tools": ["pandas", "sklearn", "plotly"],
  "expected_output": "what this step produces",
  "depends_on": []
}}

Include these task types in order: EDA → Feature Engineering → Model Selection → Evaluation → Visualization → Report.
Respond ONLY with valid JSON array."""

    response = _call_ollama(prompt, system="You are a data science task planner. Output only valid JSON.")

    try:
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        tasks = json.loads(response)
        state["analysis_plan"] = tasks
        state["current_step"] = 0
        logger.info(f"[Planner] Created plan with {len(tasks)} tasks")
    except Exception as e:
        logger.warning(f"[Planner] Task decomposition JSON parse failed: {e}")
        # Fallback default plan
        state["analysis_plan"] = _default_plan()

    return state


def create_final_plan(state: PlannerState) -> PlannerState:
    """Node: Synthesize final human-readable plan."""
    tasks_text = json.dumps(state.get("analysis_plan", []), indent=2)[:3000]

    prompt = f"""Create a clear, actionable analysis plan summary.

Objective: {state['user_objective']}
Tasks: {tasks_text}

Write a 200-word executive summary of the analysis approach, what will be done, and what insights are expected."""

    state["final_plan"] = _call_ollama(prompt)
    return state


def store_plan_to_memory(state: PlannerState) -> PlannerState:
    """Node: Persist plan to ChromaDB memory."""
    memory = AgentMemory()
    memory.store(
        content=f"Analysis Plan for: {state['user_objective']}\n\n{state['final_plan']}",
        metadata={"type": "analysis_plan", "session_id": state["session_id"]},
        doc_id=f"plan_{state['session_id']}"
    )
    for i, step in enumerate(state.get("analysis_plan", [])):
        memory.store(
            content=json.dumps(step),
            metadata={"type": "task", "step": str(i), "session_id": state["session_id"]}
        )
    return state


def build_planner_graph():
    """Build and compile the LangGraph planner workflow."""
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(PlannerState)
        graph.add_node("generate_hypotheses", generate_hypotheses)
        graph.add_node("decompose_tasks", decompose_into_tasks)
        graph.add_node("create_final_plan", create_final_plan)
        graph.add_node("store_to_memory", store_plan_to_memory)

        graph.set_entry_point("generate_hypotheses")
        graph.add_edge("generate_hypotheses", "decompose_tasks")
        graph.add_edge("decompose_tasks", "create_final_plan")
        graph.add_edge("create_final_plan", "store_to_memory")
        graph.add_edge("store_to_memory", END)

        return graph.compile()

    except ImportError:
        logger.warning("[Planner] langgraph not installed — using simple sequential planner")
        return None


def run_planner(
    user_objective: str,
    documents: list[UnifiedDocument],
    session_id: str = "default"
) -> dict:
    """
    Main entry point for the planning agent.
    Returns the complete analysis plan.
    """
    data_profiles = [doc.to_dict() for doc in documents]

    initial_state = PlannerState(
        session_id=session_id,
        user_objective=user_objective,
        data_profiles=data_profiles,
        analysis_plan=[],
        current_step=0,
        messages=[],
        hypotheses=[],
        final_plan="",
        error=""
    )

    graph = build_planner_graph()
    if graph:
        try:
            result = graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"[Planner] Graph execution failed: {e}")

    # Fallback: run nodes sequentially
    state = initial_state
    state = generate_hypotheses(state)
    state = decompose_into_tasks(state)
    state = create_final_plan(state)
    state = store_plan_to_memory(state)
    return state


def _default_plan() -> list[dict]:
    """Default analysis plan when LLM fails."""
    return [
        {"step": 1, "name": "EDA", "type": "eda", "description": "Exploratory data analysis", "tools": ["pandas", "matplotlib"], "expected_output": "Statistical summary and visualizations", "depends_on": []},
        {"step": 2, "name": "Feature Engineering", "type": "feature_engineering", "description": "Engineer features for modeling", "tools": ["pandas", "sklearn"], "expected_output": "Feature matrix", "depends_on": [1]},
        {"step": 3, "name": "Model Selection", "type": "modeling", "description": "Train and select best model", "tools": ["sklearn", "flaml"], "expected_output": "Trained model with metrics", "depends_on": [2]},
        {"step": 4, "name": "Evaluation", "type": "evaluation", "description": "Evaluate model performance", "tools": ["sklearn"], "expected_output": "Evaluation report", "depends_on": [3]},
        {"step": 5, "name": "Visualization", "type": "visualization", "description": "Create insight visualizations", "tools": ["plotly"], "expected_output": "Interactive charts", "depends_on": [4]},
        {"step": 6, "name": "Report", "type": "reporting", "description": "Generate final insight report", "tools": [], "expected_output": "Analysis report", "depends_on": [5]},
    ]
