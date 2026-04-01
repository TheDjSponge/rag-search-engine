#!/usr/bin/env python3

import argparse

from cli.semantic_search.chunked_semantic_search import ChunkedSemanticSearch
from cli.semantic_search.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
)
from cli.utils.files import load_movies
from cli.utils.text_processing import (
    chunk_sentences_semantic,
    chunk_text,
    chunk_text_with_overlap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    _ = subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generates a text embedding "
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    _ = subparsers.add_parser(
        "verify_embeddings",
        help="Builds or loads embeddings from the movies dataset",
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generates a text embedding for a user query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search",
        help="Performs semantic search to find movies related to query",
    )
    search_parser.add_argument("query", type=str, help="Query to embed")
    search_parser.add_argument(
        "--limit",
        nargs="?",
        default=5,
        type=int,
        help="Top k suggestions to show",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Performs semantic search to find movies related to query",
    )
    chunk_parser.add_argument("query", type=str, help="Query to embed")
    chunk_parser.add_argument(
        "--chunk-size",
        nargs="?",
        default=200,
        type=int,
        help="Number of expected words in a chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        nargs="?",
        default=20,
        type=int,
        help="Number of words overlapping with previous chunk",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Performs semantic search to find movies related to query",
    )
    semantic_chunk_parser.add_argument("query", type=str, help="Query to embed")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        nargs="?",
        default=4,
        type=int,
        help="Maximum number of words in a chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        nargs="?",
        default=0,
        type=int,
        help="Number of words overlapping with previous chunk",
    )

    _ = subparsers.add_parser(
        "embed_chunks",
        help="Loads movies and creates semantic chunk embeddings as well as metadata for each movie.",
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Performs semantic search to find movies related to query on chunk embeddings.",
    )
    search_chunked_parser.add_argument("query", type=str, help="Query to embed")
    search_chunked_parser.add_argument(
        "--limit",
        nargs="?",
        default=5,
        type=int,
        help="Top N matches to return.",
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
                    f"{num + 1}. {match['title']} (score: {match['score']:.4f})\n\t{match['description'][:100]} ..."
                )
        case "chunk":
            chuncked_text = chunk_command(
                args.query, args.chunk_size, args.overlap
            )
            print_chunks(chuncked_text, len(args.query))
        case "semantic_chunk":
            chunked_text = chunk_sentences_semantic(
                args.query, args.max_chunk_size, args.overlap
            )
            print_chunks(chunked_text, len(args.query))

        case "embed_chunks":
            semantic_search = ChunkedSemanticSearch()
            movies = load_movies("./data/movies.json")["movies"]
            embeddings = semantic_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
            print(f"Embeddings shape: {embeddings.shape}")

        case "search_chunked":
            semantic_search = ChunkedSemanticSearch()
            movies = load_movies("./data/movies.json")["movies"]
            embeddings = semantic_search.load_or_create_chunk_embeddings(movies)
            matched_movies = semantic_search.search_chunks(
                args.query, args.limit
            )

            for idx, semantic_match in enumerate(matched_movies):
                print(
                    f"\n{idx}. {semantic_match['title']} (score: {semantic_match['score']:.4f})"
                )
                print(f"   {semantic_match['document']}...")

        case _:
            parser.print_help()


def print_chunks(text_chunks: list[str], num_chars: int) -> None:
    print(f"Chunking {num_chars} characters")
    for id, chunk in enumerate(text_chunks):
        print(f"{id + 1}. {chunk}")


def chunk_command(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap > 0:
        return chunk_text_with_overlap(text, chunk_size, overlap)
    elif overlap == 0:
        return chunk_text(text, chunk_size)
    else:
        raise ValueError("overlap can't be a negative value.")


if __name__ == "__main__":
    main()
