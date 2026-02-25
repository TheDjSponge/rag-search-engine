from typing import List, Dict
import math
from cli.tf_idf.inverted_index import InvertedIndex


def get_sample_movies() -> List[Dict]:
    return [
        {"id": 1, "title": "Cats, cats and dogs", "description": "rule the earth"},
        {"id": 2, "title": "flat earth confirmed", "description": "in your dreams."},
        {"id": 3, "title": "bob the builder", "description": "what a dream guy"},
    ]


def get_inverted_index():
    movies = get_sample_movies()
    inverted_index = InvertedIndex()
    inverted_index.build(movies)
    return inverted_index


def idf_formula(total_doc_count, term_match_doc_count):
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def test_get_document():
    inverted_index = get_inverted_index()
    assert inverted_index.get_document("earth") == [1, 2]
    assert inverted_index.get_document("cat") == [1]
    assert inverted_index.get_document("dream") == [2, 3]


def test_get_tf():
    inverted_index = get_inverted_index()
    tf1 = inverted_index.get_tf(1, "cat")
    expected1 = 2
    assert tf1 == expected1

    tf2 = inverted_index.get_tf(2, "flats")
    expected2 = 1
    assert tf2 == expected2

    tf3 = inverted_index.get_tf(3, "nuclear")
    expected3 = 0
    assert tf3 == expected3


def test_get_idf():
    total_doc_count = 3
    inverted_index = get_inverted_index()

    idf1 = inverted_index.get_idf("dream")
    expected_idf1 = idf_formula(total_doc_count, 2)

    assert idf1 == expected_idf1

    idf2 = inverted_index.get_idf("flats")
    expected_idf2 = idf_formula(total_doc_count, 1)

    assert idf2 == expected_idf2

    idf3 = inverted_index.get_idf("nuclear")
    expected_idf3 = idf_formula(total_doc_count, 0)

    assert idf3 == expected_idf3
