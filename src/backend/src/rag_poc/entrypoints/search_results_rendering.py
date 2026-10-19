"""Renders semantic search results for the terminal."""

import textwrap
from collections.abc import Sequence

from rich.console import Console
from rich.markup import escape

from rag_poc.lib import SearchResult

_RESULT_TEXT_CHARS = 400


def render_search_results(query: str, results: Sequence[SearchResult], console: Console) -> None:
    console.rule(f"[bold]Top {len(results)} for:[/bold] {escape(query)}")
    if not results:
        console.print("No results.")
        return
    for result in results:
        rating = "n/a" if result.rating is None else f"{result.rating:g}/5"
        date = f" | {result.timestamp:%Y-%m-%d}" if result.timestamp else ""
        console.print(
            f"\n[bold]{result.rank}. score {result.score:.3f}[/bold]  "
            f"[dim]{result.feedback_id} | product {result.entity_id} | {rating}{date}[/dim]"
        )
        if result.title:
            console.print(f"[italic]{escape(result.title)}[/italic]")
        console.print(escape(textwrap.shorten(result.text, _RESULT_TEXT_CHARS, placeholder=" ...")))
