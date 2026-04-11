import argparse

from cli.hybrid_search.hybrid_search import HybridSearch
from cli.hybrid_search.normalisation import normalize
from cli.utils.constants import MOVIES_FILE_PATH
from cli.utils.files import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    normalize_parser = subparsers.add_parser(
        "normalize", help="Performs min-max normalisation on the number list"
    )
    normalize_parser.add_argument(
        "number_list",
        type=float,
        nargs="+",
        default=[],
        help="Elements to normalize",
    )

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument(
        "query", type=str, help="Query for search"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight factor for score combination. High alpha means a higher weight for keyword seach.",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of movies to recommend",
    )

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str, help="Query for search")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="A low k gives a better scores to high ranked matches, a high k flattens the scores given by ranks.",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of movies to recommend",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize(args.number_list):
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies(MOVIES_FILE_PATH)
            hybrid_search = HybridSearch(movies["movies"])
            weighted_search_matches = hybrid_search.weighted_search(
                args.query, args.alpha, args.limit
            )
            for id, match in enumerate(weighted_search_matches, start=1):
                print(f"{id}. {match[1]['title']}")
                print(f"  Hybrid Score: {match[1]['hybrid_score']}")
                print(
                    f"  BM25: {match[1]['keyword_score']}, Semantic: {match[1]['semantic_score']}"
                )
                print(f"  {match[1]['description'][:100]}")
        case "rrf-search":
            movies = load_movies(MOVIES_FILE_PATH)
            hybrid_search = HybridSearch(movies["movies"])
            rrf_search_matches = hybrid_search.rrf_search(
                args.query, args.k, args.limit
            )
            for id, rrf_match in enumerate(rrf_search_matches, start=1):
                print(f"{id}. {rrf_match[1]['title']}")
                print(f"  RRF Score: {rrf_match[1]['rrf_score']}")
                print(
                    f"  BM25 RANK: {rrf_match[1]['keyword_rank']}, Semantic Rank: {rrf_match[1]['semantic_rank']}"
                )
                print(f"  {rrf_match[1]['description'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
