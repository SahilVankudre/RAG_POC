"""Renders dataset profiles as readable terminal tables."""

import textwrap

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_poc.lib import DatasetProfile

_EXAMPLE_TEXT_CHARS = 300


def render_dataset_profile(profile: DatasetProfile, console: Console) -> None:
    console.rule(f"[bold]{profile.source_name}")
    console.print(f"Records:              {profile.record_count:,}")
    console.print(f"Reviews without text: {profile.reviews_without_text:,}")
    console.print(f"Distinct products:    {profile.unique_entities:,}")
    console.print(f"Median text length:   {profile.median_text_length:,.0f} chars")
    if profile.earliest_review and profile.latest_review:
        console.print(
            f"Review dates:         {profile.earliest_review:%Y-%m-%d} "
            f"to {profile.latest_review:%Y-%m-%d}"
        )
    if profile.raw_fields:
        console.print(f"Raw record fields:    {', '.join(profile.raw_fields)}")

    console.print(_column_table(profile))
    console.print(_rating_table(profile))
    _print_examples(profile, console)


def _column_table(profile: DatasetProfile) -> Table:
    table = Table(title="Loaded columns", title_justify="left")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Missing", justify="right")
    for column in profile.columns:
        table.add_row(column.name, column.dtype, f"{column.null_count:,}")
    return table


def _rating_table(profile: DatasetProfile) -> Table:
    table = Table(title="Rating distribution", title_justify="left")
    table.add_column("Rating", justify="right")
    table.add_column("Reviews", justify="right")
    table.add_column("Share", justify="right")
    total = sum(profile.rating_distribution.values()) or 1
    for rating, count in profile.rating_distribution.items():
        table.add_row(f"{rating:g}", f"{count:,}", f"{count / total:.1%}")
    return table


def _print_examples(profile: DatasetProfile, console: Console) -> None:
    if not profile.examples:
        return
    console.print("[bold]Sample reviews")
    for number, example in enumerate(profile.examples, start=1):
        rating = "n/a" if example.rating is None else f"{example.rating:g}/5"
        title = escape(example.title or "(no title)")
        text = escape(textwrap.shorten(example.text, _EXAMPLE_TEXT_CHARS, placeholder=" ..."))
        console.print(
            f"\n[bold]{number}. {title}[/bold]  [dim]{example.entity_id} | {rating}[/dim]"
        )
        console.print(text)
