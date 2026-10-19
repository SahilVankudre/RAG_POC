"""Interactive search session: load the model once, then answer queries until the user quits."""

from collections.abc import Sequence
from typing import Protocol

from rich.console import Console
from rich.markup import escape

from rag_poc.entrypoints.search_results_rendering import render_search_results
from rag_poc.lib import RagPocError, SearchFilters, SearchIndexManifest, SearchResult

_EXIT_COMMANDS = {"exit", "quit", ":q", "q"}


class FeedbackSearcher(Protocol):
    def warm_up(self) -> SearchIndexManifest: ...

    def search(
        self, query: str, top_k: int | None = None, filters: SearchFilters | None = None
    ) -> Sequence[SearchResult]: ...


def run_interactive_search(
    searcher: FeedbackSearcher, *, filters: SearchFilters, top_k: int | None, console: Console
) -> None:
    with console.status("Loading the search index and embedding model..."):
        manifest = searcher.warm_up()
    console.print(
        f"[bold]Semantic search[/bold] over {manifest.count:,} reviews "
        f"({manifest.model_name}). Filters: {filters.describe()}."
    )
    console.print("Type a question in plain English; [bold]exit[/bold] or Ctrl+C to quit.\n")

    while True:
        try:
            query = console.input("[bold cyan]query>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not query:
            continue
        if query.lower() in _EXIT_COMMANDS:
            return
        try:
            results = searcher.search(query, top_k=top_k, filters=filters)
        except RagPocError as error:
            console.print(f"[red]{escape(str(error))}[/red]")
            continue
        render_search_results(query, results, console)
        console.print()
