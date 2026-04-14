import argparse
import json
from typing import TypedDict

from cli.hybrid_search.hybrid_search import HybridSearch
from cli.utils.constants import GOLDEN_DS_PATH, MOVIES_FILE_PATH
from cli.utils.files import load_movies


class EvalTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    movies = load_movies(MOVIES_FILE_PATH)
    hybrid_search = HybridSearch(movies["movies"])

    dataset = load_golden_dataset(GOLDEN_DS_PATH)
    test_cases = dataset.get("test_cases", None)
    if test_cases is None:
        raise ValueError("Couldn't fetch test cases from dataset.")
    for test_case in test_cases:
        query = test_case["query"]
        print(f"- Evaluating Query: {query}")

        relevant_docs = test_case["relevant_docs"]
        matches = hybrid_search.rrf_search(query=query, k=60, limit=limit)
        matched_titles = [match[1]["title"] for match in matches]

        precision = compute_precision(matched_titles, relevant_docs, limit)
        recall = compute_recall(matched_titles, relevant_docs, limit)
        f1 = compute_f1(precision, recall)

        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}ll")
        print(f"  - F1 Score: {f1:.4f}ll")
        print(f"  - Retrieved: {matched_titles}")
        print(f"  - Relevant: {relevant_docs}")


def load_golden_dataset(file_path: str) -> dict[str, list[EvalTestCase]]:
    with open(file_path) as f:
        dataset = json.load(f)
    if type(dataset) is not dict:
        raise ValueError("Json parsing failed on the golder dataset")
    return dataset


def compute_precision(
    found_titles: list[str], labels: list[str], k: int
) -> float:
    relevant_found = set(found_titles[:k]).intersection(set(labels))
    return len(relevant_found) / len(found_titles)


def compute_recall(found_titles: list[str], labels: list[str], k: int) -> float:
    relevant_found = set(found_titles[:k]).intersection(set(labels))
    return len(relevant_found) / len(labels)


def compute_f1(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    main()
