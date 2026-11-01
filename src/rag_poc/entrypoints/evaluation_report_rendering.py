"""Renders the automatic results of an evaluation run as a terminal table."""

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_poc.lib import EvaluationRecord, EvaluationRun


def render_evaluation_run(run: EvaluationRun, console: Console) -> None:
    table = Table(title=f"Evaluation run: {run.model_name}")
    for column in ("Id", "Question", "Filters", "Outcome", "OK", "Relevance", "Cited", "Seconds"):
        table.add_column(column)
    for record in run.records:
        table.add_row(*_row(record))
    console.print(table)

    console.print(f"\nOutcome as expected: {run.as_expected}/{len(run.records)}")
    console.print(f"Failed calls: {run.failed}")
    console.print(f"Made-up citations removed: {run.invalid_citations}")
    console.print(
        "\nThis only checks answered vs refused. Read the saved run to judge whether the reviews "
        "are relevant and the citations support each claim."
    )
    console.print(f"Saved to [bold]{run.location}[/bold]")


def _row(record: EvaluationRecord) -> tuple[str, ...]:
    question = record.question
    filters = question.filters.describe() if question.filters else "default"
    identity = (question.question_id, escape(question.question), filters)
    response = record.response
    if response is None:
        return (*identity, f"[red]failed: {escape(record.error or '')}[/red]", "no", "", "", "")
    return (
        *identity,
        str(response.reason or response.status),
        "yes" if record.outcome_as_expected else "[red]no[/red]",
        "" if response.relevance is None else f"{response.relevance:.3f}",
        f"{len(response.cited)}/{len(response.retrieved)}",
        f"{response.llm.seconds:.1f}" if response.llm else "no LLM",
    )
