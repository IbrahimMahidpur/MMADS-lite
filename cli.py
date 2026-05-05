"""
CLI for the Multimodal Agentic DS Engine.

Usage:
    mmads serve              — Start the FastAPI server
    mmads ingest <file>      — Ingest a single file and print summary
    mmads run <file> <goal>  — Full analysis pipeline
    mmads memory <session>   — Show session memory
"""
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="mmads",
    help="Multimodal Agentic Data Science Engine — 100% local Ollama",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Enable hot-reload (dev mode)"),
    log_level: str = typer.Option("info", help="Uvicorn log level"),
):
    """Start the FastAPI server."""
    import uvicorn
    console.print(Panel(f"[bold green]MMADS API[/] -> http://{host}:{port}/docs", title="Starting"))
    uvicorn.run(
        "multimodal_ds.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@app.command()
def ingest(
    file: Path = typer.Argument(..., help="Path to file to ingest"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Ingest a single file and display the result."""
    from multimodal_ds.ingestion.router import route_and_ingest

    if not file.exists():
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1)

    with console.status(f"Ingesting [cyan]{file.name}[/]..."):
        doc = route_and_ingest(str(file))

    if output_json:
        print(json.dumps(doc.to_dict(), indent=2))
        return

    table = Table(title=f"Ingested: {file.name}", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", doc.id)
    table.add_row("Type", doc.data_type.value)
    table.add_row("Status", f"[green]{doc.status.value}[/]" if doc.status.value == "done" else f"[red]{doc.status.value}[/]")
    table.add_row("Processor", doc.provenance.processor)
    table.add_row("Time (s)", str(doc.provenance.processing_time_s))
    table.add_row("Text preview", doc.text_content[:200] + "..." if doc.text_content else "—")
    if doc.schema_info:
        table.add_row("Shape", str(doc.schema_info.get("shape", "—")))
        table.add_row("Columns", str(len(doc.schema_info.get("columns", []))))
    console.print(table)


@app.command()
def run(
    files: list[Path] = typer.Argument(..., help="One or more files to analyse"),
    objective: str = typer.Option(..., "--objective", "-o", help="Analysis goal"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID"),
    max_tasks: int = typer.Option(6, "--max-tasks", help="Max tasks to execute"),
    no_stats: bool = typer.Option(False, "--no-stats", help="Skip statistical checks"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Run the full agentic analysis pipeline on one or more files."""
    from multimodal_ds.agents.orchestrator import AgentOrchestrator

    missing = [f for f in files if not f.exists()]
    if missing:
        console.print(f"[red]Files not found:[/] {missing}")
        raise typer.Exit(1)

    orchestrator = AgentOrchestrator()
    console.print(Panel(f"[bold]Objective:[/] {objective}\n[bold]Files:[/] {[f.name for f in files]}", title="[green]MMADS Run[/]"))

    with console.status("Running pipeline..."):
        result = orchestrator.run(
            file_paths=[str(f) for f in files],
            objective=objective,
            session_id=session_id,
            run_statistical_checks=not no_stats,
            max_tasks=max_tasks,
        )

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    d = result.to_dict()
    status_color = "green" if d["status"] == "success" else ("yellow" if d["status"] == "partial" else "red")
    console.print(Panel(
        f"Status: [{status_color}]{d['status']}[/]\n"
        f"Tasks: {d['tasks_completed']}/{d['tasks_planned']} completed\n"
        f"Files created: {len(d['files_created'])}\n"
        f"Duration: {d['duration_s']}s\n"
        f"Session: {d['session_id']}",
        title="Result"
    ))

    if d.get("plan_summary"):
        console.print(Panel(d["plan_summary"], title="Plan Summary"))

    if d.get("errors"):
        console.print(Panel("\n".join(d["errors"]), title="[red]Errors[/]"))

    if d.get("files_created"):
        console.print("\n[bold]Generated files:[/]")
        for f in d["files_created"]:
            console.print(f"  * {f}")


@app.command()
def memory(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    n: int = typer.Option(10, "--n", help="Number of entries to show"),
):
    """Display stored memory entries for a session."""
    from multimodal_ds.memory.agent_memory import AgentMemory

    mem = AgentMemory()
    entries = mem.get_session_history(session_id)[:n]

    if not entries:
        console.print(f"[yellow]No memory found for session:[/] {session_id}")
        return

    for i, entry in enumerate(entries, 1):
        meta = entry.get("metadata", {})
        console.print(Panel(
            entry["content"][:400],
            title=f"[{i}] step={meta.get('step', '?')} | {meta.get('timestamp', '')}",
        ))


if __name__ == "__main__":
    app()
