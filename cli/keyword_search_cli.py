#!/usr/bin/env python3

import argparse
from typing import List, Dict
from utils.files import load_movies
from utils.search import (
    find_matching_movies,
    print_matched_movies,
    find_matching_movies_with_index,
)
from tf_idf.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="Builds the inverted index of movies.")

    tf_parser = subparsers.add_parser(
        "tf", help="Shows the term frequecy (TF) of a term for a given document id"
    )
    tf_parser.add_argument("doc_id", type=int, help="id of the document")
    tf_parser.add_argument("term", type=str, help="term to search in the document")

    idf_parser = subparsers.add_parser(
        "idf",
        help="Shows the inverse document frequency (IDF) of a term over the dataset",
    )
    idf_parser.add_argument(
        "term", type=str, help="Term for which to show the inverse document frequency"
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Shows the TF-IDF score of a word in a given document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="id of the document")
    tfidf_parser.add_argument("term", type=str, help="term to search in the document")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
