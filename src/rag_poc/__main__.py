"""Allows ``python -m rag_poc`` as an alternative to the ``rag-poc`` console script."""

from rag_poc.entrypoints.cli import run

if __name__ == "__main__":
    run()
