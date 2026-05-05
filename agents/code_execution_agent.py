"""
Code Execution Agent — writes and runs pandas/sklearn/plotly code in sandbox.
Specialist agent #1 of 3.
"""
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from multimodal_ds.config import CODER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class CodeExecutionAgent:
    """
    Specialist agent that:
    1. Receives a task description + data context
    2. Generates Python code using Ollama
    3. Executes in sandboxed subprocess
    4. Returns results + stores in memory
    """

    SYSTEM_PROMPT = """You are an expert Python data scientist.
Write clean, self-contained Python code that:
- Uses pandas, numpy, sklearn, scipy, plotly, matplotlib, seaborn as needed
- Saves ALL outputs (plots, CSVs, models) to the current directory
- Uses matplotlib.use('Agg') before importing pyplot
- NEVER calls plt.show() — only plt.savefig()
- Loads data using the EXACT filenames provided in the Data Context (e.g., pd.read_csv('filename.csv'))
- Handles all exceptions gracefully
- Prints a summary at the end

Output ONLY the Python code inside ```python ... ``` fences. Nothing else."""

    def __init__(self, working_dir: Optional[str] = None, session_id: str = "default"):
        self.working_dir = Path(working_dir or OUTPUT_DIR)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.memory = AgentMemory()

    def execute_task(self, task: dict, data_context: str = "", max_retries: int = 2) -> dict:
        """
        Execute a single analysis task.
        Returns dict with: success, code, output, files_created, error
        """
        task_desc = task.get("description", str(task))
        task_name = task.get("name", "task")

        logger.info(f"[CodeAgent] Executing: {task_name}")

        # Retrieve relevant memory
        past_context = self._get_relevant_memory(task_desc)

        # Generate code
        code = self._generate_code(task_desc, data_context, past_context)
        if not code:
            return {"success": False, "error": "Code generation failed", "code": "", "output": "", "files_created": []}

        # Execute with retries
        result = self._execute_with_retry(code, task_desc, data_context, max_retries)

        # Store result in memory
        status_msg = "successfully" if result["success"] else "with errors"
        self.memory.store_analysis_step(
            step_name=task_name,
            result=(
                f"Code executed {status_msg}.\n"
                f"Output: {result['output'][:500]}\n"
                f"Files: {result['files_created']}\n"
                f"Error: {result.get('error', '')}\n\n"
                f"Code used:\n```python\n{result.get('code', '')}\n```"
            ),
            session_id=self.session_id
        )

        return result

    def _generate_code(self, task_desc: str, data_context: str, past_context: str) -> str:
        """Generate Python code for the task using Ollama."""
        import httpx

        prompt = f"""Task: {task_desc}

Data Context:
{data_context[:1500]}

Previous Analysis Context:
{past_context[:500]}

Working directory: {self.working_dir}

Write Python code to complete this task. Save all outputs to the current directory.
Use simple filenames without subdirectories (e.g., 'plot.png' not 'output/plot.png')."""

        model = CODER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"num_predict": 2000, "temperature": 0.1},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "")
                return self._extract_code(content)
        except Exception as e:
            logger.error(f"[CodeAgent] Code generation failed: {e}")
        return ""

    def _execute_code(self, code: str) -> tuple[bool, str, list[str]]:
        """Execute Python code in subprocess, return (success, output, files_created)."""
        files_before = set(self.working_dir.glob("*"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py",
            dir=self.working_dir, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]:\n{result.stderr}"
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            return False, "Execution timed out after 300s", []
        except Exception as e:
            return False, f"Execution error: {e}", []
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass

        files_after = set(self.working_dir.glob("*"))
        new_files = [f.name for f in (files_after - files_before) if f.is_file()]

        return success, output, new_files

    def _execute_with_retry(self, code: str, task_desc: str, data_context: str, max_retries: int) -> dict:
        """Execute code, retry with fix if fails."""
        success, output, files = self._execute_code(code)

        if success:
            return {"success": True, "code": code, "output": output, "files_created": files, "error": ""}

        # Try to fix
        for attempt in range(max_retries):
            logger.info(f"[CodeAgent] Attempt {attempt + 1} fix...")
            fix_code = self._generate_fix(code, output, task_desc)
            if fix_code:
                success, output, files = self._execute_code(fix_code)
                if success:
                    return {"success": True, "code": fix_code, "output": output, "files_created": files, "error": ""}
                code = fix_code  # Use latest code for next fix

        return {"success": False, "code": code, "output": output, "files_created": files, "error": output}

    def _generate_fix(self, failed_code: str, error_output: str, task_desc: str) -> str:
        """Generate a fix for failing code."""
        import httpx
        model = CODER_MODEL.replace("ollama/", "")
        prompt = f"""Fix this Python code that failed.

Original task: {task_desc}

Failed code:
```python
{failed_code[:1500]}
```

Error:
{error_output[:500]}

Provide ONE complete fixed Python script in ```python ... ``` fences.
Use only simple filenames, no subdirectories. Do not use !pip or bash commands."""

        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Fix Python code. Output only the fixed code in ```python``` fences."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"num_predict": 2000, "temperature": 0.1},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "")
                return self._extract_code(content)
        except Exception:
            pass
        return ""

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown fences."""
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()

    def _get_relevant_memory(self, query: str) -> str:
        """Retrieve relevant past analysis from memory."""
        memories = self.memory.retrieve(query, n_results=3)
        if not memories:
            return ""
        return "\n".join(m["content"][:200] for m in memories)
