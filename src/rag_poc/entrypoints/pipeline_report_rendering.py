"""Renders the stage-by-stage summary of a full index build."""

from rich.console import Console
from rich.table import Table

from rag_poc.lib import PipelineReport


def render_pipeline_report(report: PipelineReport, console: Console) -> None:
    table = Table(title="Build summary", title_justify="left")
    table.add_column("Stage")
    table.add_column("Result")
    table.add_column("Status")

    quality = report.quality
    embedding = report.embedding
    index = report.index
    table.add_row(
        "1. Sample",
        f"{report.sample.profile.record_count:,} reviews ({report.sample.location.name})",
        _status(reused=not report.sample.created),
    )
    table.add_row(
        "2. Prepare",
        f"{quality.prepared_records:,} prepared, {quality.excluded_records:,} excluded",
        "[green]reconciled" if quality.is_reconciled else "[red]NOT reconciled",
    )
    table.add_row(
        "3. Embed",
        f"{embedding.count:,} x {embedding.dimension} ({embedding.model_name})",
        _status(reused=embedding.reused, seconds=embedding.seconds),
    )
    table.add_row(
        "4. Index",
        f"{index.manifest.index_type}, {index.manifest.count:,} vectors",
        _status(reused=index.reused, seconds=index.seconds),
    )
    console.print(table)
    console.print(f"Done in {report.seconds:.1f}s. Index: [bold]{index.location}[/bold]")


def _status(*, reused: bool, seconds: float | None = None) -> str:
    if reused:
        return "reused (up to date)"
    return "built" if seconds is None else f"built in {seconds:.1f}s"
