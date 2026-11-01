"""Renders a RAG answer for the terminal: question, sources, optional context and the answer."""

import textwrap

from rich.console import Console
from rich.markup import escape

from rag_poc.lib import RAGResponse, RAGSource

_RETRIEVED_TEXT_CHARS = 200
_CITED_TEXT_CHARS = 300


def render_rag_response(response: RAGResponse, console: Console, *, show_context: bool) -> None:
    console.rule("[bold]Question[/bold]")
    console.print(escape(response.question))

    relevance = "" if response.relevance is None else f", relevance {response.relevance:.3f}"
    console.rule(f"[bold]Retrieved reviews[/bold] ({len(response.retrieved)}{relevance})")
    for source in response.retrieved:
        _render_retrieved(source, console)

    if show_context and response.context:
        console.rule("[bold]Context sent to the LLM[/bold]")
        console.print(escape(response.context))

    outcome = f"{response.status}: {response.reason}" if response.reason else response.status
    console.rule(f"[bold]Answer[/bold] [dim]({outcome})[/dim]")
    console.print(escape(response.answer))

    if response.cited:
        console.rule(f"[bold]Supporting reviews[/bold] ({len(response.cited)})")
        for source in response.cited:
            _render_cited(source, console)
    _render_warnings(response, console)


def _render_retrieved(source: RAGSource, console: Console) -> None:
    review = source.review
    console.print(
        f"[bold]{source.label}[/bold] score {review.score:.3f}  "
        f"[dim]{review.feedback_id} | product {review.entity_id} | {_rating(source)}[/dim]"
    )
    console.print(f"   {escape(_excerpt(review.text, _RETRIEVED_TEXT_CHARS))}")


def _render_cited(source: RAGSource, console: Console) -> None:
    review = source.review
    console.print(
        f"[bold]{source.label}[/bold]  Review {review.feedback_id}  ·  "
        f"Rating: {_rating(source)}  ·  [dim]product {review.entity_id}[/dim]"
    )
    console.print(f'   "{escape(_excerpt(review.text, _CITED_TEXT_CHARS))}"')


def _render_warnings(response: RAGResponse, console: Console) -> None:
    if response.invalid_citations:
        labels = ", ".join(response.invalid_citations)
        console.print(
            f"\n[yellow]Removed citations of reviews that were not retrieved: "
            f"{escape(labels)}. Claims that relied on them are unsupported.[/yellow]"
        )
    if response.llm is None:
        return
    llm = response.llm
    if not llm.is_complete:
        console.print(f"\n[yellow]The answer stopped early ({escape(llm.finish_reason)}).[/yellow]")
    console.print(
        f"\n[dim]{escape(llm.model_name)} | {llm.seconds:.1f}s | "
        f"{llm.input_tokens} tokens in, {llm.output_tokens} out[/dim]"
    )


def _rating(source: RAGSource) -> str:
    rating = source.review.rating
    return "n/a" if rating is None else f"{rating:g}/5"


def _excerpt(text: str, max_chars: int) -> str:
    return textwrap.shorten(text, max_chars, placeholder=" ...")
