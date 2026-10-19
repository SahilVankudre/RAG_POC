"""Command-line interface — the application's user-facing entry point.

Each command obtains collaborators from the composition root (``AppContainer``) and never
constructs adapters itself.
"""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rag_poc.bootstrap import AppContainer, build_container, load_settings
from rag_poc.entrypoints.data_quality_report_rendering import render_data_quality_report
from rag_poc.entrypoints.dataset_profile_rendering import render_dataset_profile
from rag_poc.entrypoints.interactive_search import run_interactive_search
from rag_poc.entrypoints.pipeline_report_rendering import render_pipeline_report
from rag_poc.entrypoints.search_results_rendering import render_search_results
from rag_poc.lib import RagPocError, SearchFilters, configure_logging

ExampleCount = Annotated[int, typer.Option("--examples", min=0, help="Sample reviews to show.")]

app = typer.Typer(
    name="rag-poc",
    help="Semantic search over customer feedback (Amazon reviews -> embeddings -> FAISS).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def configure_app(
    ctx: typer.Context,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = None,
) -> None:
    """Build the composition root once per invocation and share it with every command."""
    overrides = {"log_level": log_level} if log_level else {}
    container = build_container(load_settings(**overrides))
    configure_logging(container.settings.log_level)
    ctx.obj = container


@app.command("show-config")
def show_config(ctx: typer.Context) -> None:
    """Print the resolved settings after .env and RAG_* environment overrides."""
    container: AppContainer = ctx.obj
    settings = container.settings
    resolved = settings.model_dump() | {
        "raw_data_dir": settings.raw_data_dir,
        "review_sample_path": settings.review_sample_path,
        "prepared_feedback_path": settings.prepared_feedback_path,
        "quality_report_path": settings.quality_report_path,
        "index_dir": settings.index_dir,
        "embeddings_path": settings.embeddings_path,
        "faiss_index_path": settings.faiss_index_path,
    }
    width = max(len(name) for name in resolved)
    for name, value in resolved.items():
        typer.echo(f"{name:<{width}}  {value}")


@app.command("explore")
def explore_dataset(ctx: typer.Context, examples: ExampleCount = 3) -> None:
    """Load the full raw review dataset and show its fields, size and sample reviews."""
    container: AppContainer = ctx.obj
    profile = container.dataset_exploration.explore(example_count=examples)
    render_dataset_profile(profile, Console())


@app.command("sample")
def create_review_sample(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Recreate the sample even if it already exists.")
    ] = False,
    examples: ExampleCount = 3,
) -> None:
    """Draw the seeded review sample, save it as Parquet and summarize it."""
    container: AppContainer = ctx.obj
    report = container.review_sample_creation.create_sample(force=force, example_count=examples)
    console = Console()
    render_dataset_profile(report.profile, console)
    action = "Saved new sample to" if report.created else "Using existing sample at"
    console.print(f"\n{action} [bold]{report.location}[/bold]")


@app.command("prepare")
def prepare_feedback(ctx: typer.Context) -> None:
    """Convert the sample to FeedbackItems, clean it and save it with a data-quality report."""
    container: AppContainer = ctx.obj
    report = container.feedback_preparation.prepare()
    console = Console()
    render_data_quality_report(report, console)
    store = container.processed_feedback_store
    console.print(f"\nSaved prepared feedback to [bold]{store.location}[/bold]")
    console.print(f"Saved data-quality report to [bold]{store.report_location}[/bold]")


@app.command("embed")
def embed_feedback(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Re-embed even if stored embeddings are up to date.")
    ] = False,
) -> None:
    """Embed every prepared feedback item in batches and save the vectors."""
    container: AppContainer = ctx.obj
    console = Console()
    with _embedding_progress(console) as on_progress:
        report = container.feedback_embedding.embed_feedback(force=force, on_progress=on_progress)

    if report.reused:
        console.print(f"Embeddings already up to date at [bold]{report.location}[/bold]")
    else:
        rate = report.count / report.seconds if report.seconds else 0.0
        console.print(
            f"Embedded {report.count:,} texts with {report.model_name} in {report.seconds:.1f}s "
            f"({rate:,.0f}/s, batch size {report.batch_size})"
        )
        console.print(f"Saved to [bold]{report.location}[/bold]")
    console.print(f"Vectors: {report.count:,} x {report.dimension} (float32, L2-normalized)")


@app.command("index")
def build_search_index(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if the index is up to date.")
    ] = False,
) -> None:
    """Build the FAISS index from stored embeddings and save it with its id map."""
    container: AppContainer = ctx.obj
    report = container.search_index_building.build_index(force=force)
    manifest = report.manifest
    console = Console()
    if report.reused:
        console.print(f"Search index already up to date at [bold]{report.location}[/bold]")
    else:
        console.print(f"Built {manifest.index_type} over {manifest.count:,} vectors")
        console.print(f"in {report.seconds:.2f}s and saved to [bold]{report.location}[/bold]")
    console.print(f"Model: {manifest.model_name} ({manifest.dimension}-d, cosine similarity)")


@app.command("build-index")
def build_index_pipeline(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Recompute embeddings and rebuild the index.")
    ] = False,
) -> None:
    """Run every stage: sample -> prepare -> embed -> index (each reused when up to date)."""
    container: AppContainer = ctx.obj
    console = Console()
    with _embedding_progress(console) as on_progress:
        report = container.search_index_pipeline.run(force=force, on_embedding_progress=on_progress)
    render_pipeline_report(report, console)


@app.command("search")
def search_feedback(
    ctx: typer.Context,
    query: Annotated[
        str | None,
        typer.Argument(help="Natural-language query. Omit it for an interactive session."),
    ] = None,
    top_k: Annotated[
        int | None, typer.Option("--top-k", "-k", min=1, help="Number of results.")
    ] = None,
    min_rating: Annotated[
        float | None, typer.Option("--min-rating", min=0, help="Only reviews rated at least this.")
    ] = None,
    max_rating: Annotated[
        float | None,
        typer.Option(
            "--max-rating", min=0, help="Only reviews rated at most this (2 = complaints)."
        ),
    ] = None,
    min_words: Annotated[
        int | None,
        typer.Option(
            "--min-words",
            min=0,
            help="Skip reviews with fewer words (default: RAG_SEARCH_MIN_WORDS, 3).",
        ),
    ] = None,
) -> None:
    """Find the reviews most semantically similar to QUERY (no LLM involved)."""
    container: AppContainer = ctx.obj
    searcher = container.feedback_search
    try:
        filters = SearchFilters(
            min_rating=min_rating,
            max_rating=max_rating,
            min_words=searcher.default_filters.min_words if min_words is None else min_words,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    console = Console()
    if query is None:
        run_interactive_search(searcher, filters=filters, top_k=top_k, console=console)
        return
    results = searcher.search(query, top_k=top_k, filters=filters)
    render_search_results(query, results, console)


@contextmanager
def _embedding_progress(console: Console) -> Iterator[Callable[[int, int], None]]:
    """Progress bar for embedding; yields the callback that use cases report progress to."""
    with Progress(
        TextColumn("Embedding"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("embed", total=None)
        yield lambda done, total: progress.update(task, completed=done, total=total)


def run() -> None:
    """Console-script entry point: runs the CLI and reports expected errors without a traceback."""
    _use_utf8_console()
    try:
        app()
    except RagPocError as error:
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from error


def _use_utf8_console() -> None:
    """Review text has emoji and non-Latin scripts that legacy Windows code pages can't print."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
