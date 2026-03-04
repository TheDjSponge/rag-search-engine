#!/usr/bin/env python3

import argparse
from cli.semantic_search.semantic_search import SemanticSearch
from cli.semantic_search.semantic_search import (
    embed_text,
    verify_embeddings,
    embed_query_text,
)
from cli.utils.files import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generates a text embedding "
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    _ = subparsers.add_parser(
        "verify_embeddings", help="Builds or loads embeddings from the movies dataset"
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generates a text embedding for a user query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Performs semantic search to find movies related to query"
    )
    search_parser.add_argument("query", type=str, help="Query to embed")
    search_parser.add_argument(
        "--limit", nargs="?", default=5, type=int, help="Top k suggestions to show"
    )

    args = parser.parse_args()
    match args.command:

        case "verify":
            semantic_search = SemanticSearch()
            semantic_search.verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            movies = load_movies("./data/movies.json")["movies"]
            semantic_search.load_or_create_embeddings(movies)
            matches = semantic_search.search(args.query, args.limit)
            for num, match in enumerate(matches):
                print(
                    f"{num+1}. {match["title"]} (score: {match["score"]:.4f})\n\t{match["description"][:100]} ..."
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
