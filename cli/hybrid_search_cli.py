import argparse

from cli.hybrid_search.hybrid_search import HybridSearch
from cli.hybrid_search.normalisation import normalize
from cli.llm_shenanigans.query_enhancement import (
    expand_query,
    rewrite_query,
    spell_checking,
)
from cli.llm_shenanigans.reranking import (
    evaluate,
    rerank_batch,
    rerank_movies,
    rerank_with_cross_encoder,
)
from cli.utils.constants import MOVIES_FILE_PATH
from cli.utils.files import load_movies
from cli.utils.models import RRFMatchItem


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
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )
    rrf_search_parser.add_argument("--debug", type=str, help="A title to track")
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="A title to track",
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
            if args.debug:
                print(f"Debbuging for movie {args.debug}")
            movies = load_movies(MOVIES_FILE_PATH)
            hybrid_search = HybridSearch(movies["movies"])

            if args.rerank_method in ["individual", "batch", "cross_encoder"]:
                limit = 5 * args.limit
            else:
                limit = args.limit

            if args.enhance and "spell" in args.enhance:
                query = spell_checking(args.query)
            elif args.enhance and "rewrite" in args.enhance:
                query = rewrite_query(args.query)
            elif args.enhance and "expand" in args.enhance:
                query = expand_query(args.query)
            else:
                query = args.query

            rrf_search_matches = hybrid_search.rrf_search(
                query, args.k, limit, debug=args.debug
            )
            if args.debug:
                debug_query_position = -1
                for id, rrf_match_elem in enumerate(rrf_search_matches):
                    if args.debug in rrf_match_elem[1]["title"]:
                        debug_query_position = id
                        break

                print(
                    f"DEBUG[RRF_SEARCH]: The searched movie {args.debug} is at position {debug_query_position}"
                )

            matched_movies = [match[1] for match in rrf_search_matches]
            if args.rerank_method == "individual":
                matched_movies = rerank_movies(
                    query, matched_movies, limit // 5
                )
            elif args.rerank_method == "batch":
                matched_movies = rerank_batch(query, matched_movies, limit // 5)
            elif args.rerank_method == "cross_encoder":
                matched_movies = rerank_with_cross_encoder(
                    query, matched_movies, limit // 5, args.debug
                )

            if args.debug:
                debug_query_position = -1
                for id, elem in enumerate(matched_movies):
                    if args.debug in elem["title"]:
                        debug_query_position = id
                        break

                print(
                    f"DEBUG[Reranking]: The searched movie {args.debug} is at position {debug_query_position}"
                )

            for id, rrf_match in enumerate(matched_movies, start=1):
                print(f"{id}. {rrf_match['title']}")
                if "rerank_score" in rrf_match:
                    print(
                        f"  Re-Rank Score: {rrf_match['rerank_score']:.3f}/10"
                    )
                if "cross_encoder_score" in rrf_match:
                    print(
                        f"  Cross Encoder Score: {rrf_match['cross_encoder_score']:.3f}"
                    )
                print(f"  RRF Score: {rrf_match['rrf_score']}")
                print(
                    f"  BM25 RANK: {rrf_match['keyword_rank']}, Semantic Rank: {rrf_match['semantic_rank']}"
                )
                print(f"  {rrf_match['description'][:100]}")

            if args.evaluate:
                evals = evaluate(query, matched_movies)
                format_evaluation(matched_movies, evals)
        case _:
            parser.print_help()


def format_evaluation(
    candidate_movies: list[RRFMatchItem], evaluation_scores: list[int]
) -> None:
    print("--------- EVALUATION REPORT ------------")
    for idx, (movie, score) in enumerate(
        zip(candidate_movies, evaluation_scores, strict=True)
    ):
        print(f"{idx}. {movie['title']}: {score}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
