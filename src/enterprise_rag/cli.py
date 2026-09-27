"""Command-line entry point: ``rag init-index | ingest | query``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .composition import RagServices, build_services
from .domain.errors import RagError

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Enterprise Modular Research-Paper RAG POC (Phase 1: Naive RAG).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-index", help="Create/update the Azure AI Search index.")

    ingest = subparsers.add_parser("ingest", help="Ingest documents from a file or directory.")
    ingest.add_argument("path", help="Path to a file or a directory of documents.")

    query = subparsers.add_parser("query", help="Ask a question against the index.")
    query.add_argument("question", help="The question to answer.")

    return parser


def _run(args: argparse.Namespace, services: RagServices) -> int:
    if args.command == "init-index":
        ensure_index = getattr(services.store, "ensure_index", None)
        if ensure_index is None:
            print("The configured vector store does not support index creation.")
            return 1
        ensure_index()
        print("Index is ready.")
        return 0

    if args.command == "ingest":
        documents = services.loader.load(args.path)
        result = services.ingestion.ingest(documents)
        print(f"Ingested {result.documents} document(s) into {result.chunks} chunk(s).")
        return 0

    if args.command == "query":
        answer = services.query.answer(args.question)
        print(answer.answer)
        if answer.sources:
            print("\nSources:")
            for source in answer.sources:
                print(f"- {source.chunk.id} (score={source.score:.3f})")
        return 0

    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        services = build_services()
        return _run(args, services)
    except RagError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
