import argparse
from cli.utils.files import load_movies
from cli.utils.models import RRFMatchItem
from cli.utils.constants import MOVIES_FILE_PATH
from cli.hybrid_search.hybrid_search import HybridSearch
from cli.llm_shenanigans.llm_utils import make_llm_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
            # do RAG stuff here
        case _:
            parser.print_help()


def rag_command(query: str, limit: int = 5):
    movies = load_movies(MOVIES_FILE_PATH)["movies"]
    hybrid_search = HybridSearch(movies)

    matches = hybrid_search.rrf_search(query, k=60, limit=5, debug=None)
    match_items = [match[1] for match in matches]
    display_search_results(match_items)
    doc_list = ""
    for match in candidate_matches:
        doc_list += json.dumps(match)


def display_search_results(searched_entries: list[RRFMatchItem]) -> None:
    print("Search Results:")
    for entry in searched_entries:
        print(f"- {entry["title"]}")


if __name__ == "__main__":
    main()
