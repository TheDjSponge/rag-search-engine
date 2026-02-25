from typing import List, Dict
from cli.utils.search import (
    has_token_intersection,
    find_matching_movies,
    find_matching_movies_with_index,
)
from cli.tf_idf.inverted_index import InvertedIndex


def get_sample_movies() -> List[Dict]:
    return [
        {"id": 1, "title": "Cats, cats and dogs", "description": "rule the earth"},
        {"id": 2, "title": "flat earth confirmed", "description": "in your dreams."},
        {"id": 3, "title": "bob the builder", "description": "what a dream guy"},
    ]


def test_token_intersection():
    set1 = ["yes", "no"]
    set2 = ["no", "maybe"]
    set3 = ["lol"]
    assert has_token_intersection(set1, set2)
    assert not has_token_intersection(set1, set3)


def test_find_matching_movies():
    movies = get_sample_movies()

    query1 = "cats"
    matches1 = find_matching_movies(movies, query1)
    expected1 = ["Cats, cats and dogs"]
    assert matches1 == expected1

    query2 = "nuclear"
    matches2 = find_matching_movies(movies, query2)
    expected2 = []
    assert matches2 == expected2


def test_find_matching_movies_with_index():
    movies = get_sample_movies()
    inverted_index = InvertedIndex()
    inverted_index.build(movies)

    query1 = "cats"
    matches1 = find_matching_movies_with_index(query1, movies, inverted_index)
    expected1 = ["Cats, cats and dogs"]
    assert matches1 == expected1

    query2 = "nuclear"
    matches2 = find_matching_movies_with_index(query2, movies, inverted_index)
    expected2 = []
    assert matches2 == expected2

    ## in comparison to the base matching, here words also match on descriptions
    query3 = "dream"
    matches3 = find_matching_movies_with_index(query3, movies, inverted_index)
    expected3 = ["flat earth confirmed", "bob the builder"]
    assert matches3 == expected3
