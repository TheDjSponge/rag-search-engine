from typing import Dict, List
from .files import load_stopwords
from cli.tf_idf.inverted_index import InvertedIndex
from .text_processing import prepare_and_tokenize


def find_matching_movies(movies: List[Dict], query: str) -> List[str]:

    matched = []
    prepared_query = prepare_and_tokenize(query)

    for movie in movies:
        prepared_title = prepare_and_tokenize(movie["title"])

        if has_token_intersection(prepared_query, prepared_title):
            matched.append(movie["title"])

    return matched


def find_matching_movies_with_index(
    query: str, movies: List[Dict], invertedIndex: InvertedIndex
) -> List[str]:

    matched_ids = []
    prepared_query = prepare_and_tokenize(query)

    for token in prepared_query:
        matched_ids += invertedIndex.get_document(token)
        if len(matched_ids) >= 5:
            break

    matched = []
    for movie_id in matched_ids[:5]:
        matched.append(movies[movie_id - 1]["title"])

    return matched


def print_matched_movies(matched_list: List[str]) -> None:
    print("\n--- Words found through direct matching ---\n")
    for movie_title in matched_list:
        print(f"- {movie_title}")


def has_token_intersection(tokens_1: List[str], tokens_2: List[str]) -> bool:
    return len(set(tokens_1).intersection(set(tokens_2))) > 0
