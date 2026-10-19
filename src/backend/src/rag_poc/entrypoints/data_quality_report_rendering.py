"""Renders the data-quality report as a terminal summary."""

from rich.console import Console
from rich.table import Table

from rag_poc.lib import DataQualityReport


def render_data_quality_report(report: DataQualityReport, console: Console) -> None:
    console.rule(f"[bold]Data quality - {report.source_name}")

    flow = Table(title="Record flow", title_justify="left")
    flow.add_column("Stage")
    flow.add_column("Records", justify="right")
    flow.add_row("Sampled reviews", f"{report.input_records:,}")
    for reason, count in sorted(report.adapter_rejections.items()):
        flow.add_row(f"  - rejected by adapter: {reason}", f"-{count:,}")
    for reason, count in sorted(report.cleaning_exclusions.items()):
        flow.add_row(f"  - excluded by cleaning: {reason}", f"-{count:,}")
    flow.add_row("[bold]Prepared for embedding", f"[bold]{report.prepared_records:,}")
    console.print(flow)

    changes = Table(title="Text changes (records affected)", title_justify="left")
    changes.add_column("Change")
    changes.add_column("Records", justify="right")
    for kind, count in sorted(report.text_modifications.items()):
        changes.add_row(kind.replace("_", " "), f"{count:,}")
    changes.add_row("title combined into embedding text", f"{report.titles_combined:,}")
    console.print(changes)

    console.print(
        f"Kept but notable: {report.short_text_records:,} very short texts, "
        f"{report.long_text_records:,} long texts the model will truncate."
    )
    status = "[green]reconciled" if report.is_reconciled else "[red]NOT reconciled"
    console.print(f"Accounting: {status}[/] ({report.excluded_records:,} excluded in total)")
