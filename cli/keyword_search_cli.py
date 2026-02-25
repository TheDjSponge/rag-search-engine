#!/usr/bin/env python3

import argparse
from typing import List, Dict
from utils.files import load_movies
from utils.search import find_matching_movies, print_matched_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies = load_movies("./data/movies.json")
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            direct_matched_movies = find_matching_movies(movies, args.query)
            print_matched_movies(direct_matched_movies)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
