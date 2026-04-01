#!/usr/bin/env python3

import argparse

from cli.tf_idf.inverted_index import InvertedIndex
from cli.utils.constants import BM25_B, BM25_K1
from cli.utils.files import load_movies
from cli.utils.models import BM25Match, MovieEntry
from cli.utils.search import (
    find_matching_movies_with_index,
    print_matched_movies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25"
    )
    search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser(
        "build", help="Builds the inverted index of movies."
    )

    tf_parser = subparsers.add_parser(
        "tf",
        help="Shows the term frequecy (TF) of a term for a given document id",
    )
    tf_parser.add_argument("doc_id", type=int, help="id of the document")
    tf_parser.add_argument(
        "term", type=str, help="term to search in the document"
    )

    idf_parser = subparsers.add_parser(
        "idf",
        help="Shows the inverse document frequency (IDF) of a term over the dataset",
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Term for which to show the inverse document frequency",
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Shows the TF-IDF score of a word in a given document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="id of the document")
    tfidf_parser.add_argument(
        "term", type=str, help="term to search in the document"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )
    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=BM25_B,
        help="Tunable BM25 b parameter",
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies = load_movies("./data/movies.json")
    inverted_index = InvertedIndex()
    match args.command:
        case "search":
            # print(f"Searching for: {args.query}")
            # direct_matched_movies = find_matching_movies(movies, args.query)

            inverted_index.load()
            matches = find_matching_movies_with_index(
                args.query, movies["movies"], inverted_index
            )
            print_matched_movies(matches)

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build(movies["movies"])
            inverted_index.save()

        case "tf":
            inverted_index.load()
            term_frequency = inverted_index.get_tf(args.doc_id, args.term)
            print(f"The term '{args.term}' appears {term_frequency} time(s)")

        case "idf":
            inverted_index.load()
            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            idf = inverted_index.get_idf(args.term)
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}"
            )
        case "bm25search":
            bm25_matches = bm25search_command(args.query)
            print_bm25_matches(bm25_matches, movies["movies"])
        case _:
            parser.print_help()


def bm25search_command(query: str) -> list[BM25Match]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    matches = inverted_index.bm25_search(query)
    return matches


def print_bm25_matches(
    bm25_matches: list[BM25Match], movies: list[MovieEntry]
) -> None:
    for match in bm25_matches:
        print(
            f"({match['doc_id']}) {movies[match['doc_id'] - 1]['title']} - Score: {match['bm25_score']:.2f}"
        )


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    bm25_idf = inverted_index.get_bm25_idf(term)
    return bm25_idf


def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    bm25_tf = inverted_index.get_bm25_tf(doc_id, term, k1, b)

    return bm25_tf


if __name__ == "__main__":
    main()
