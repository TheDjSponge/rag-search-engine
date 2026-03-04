import numpy as np
import pytest

from cli.semantic_search.semantic_search import (
    SemanticSearch,
    cosine_similarity,
)
from cli.utils.models import MovieEntry


def get_sample_movies() -> list[MovieEntry]:
    return [
        {
            "id": 1,
            "title": "Cats, cats and dogs",
            "description": "rule the earth",
        },
        {
            "id": 2,
            "title": "flat earth confirmed",
            "description": "in your dreams.",
        },
        {
            "id": 3,
            "title": "bob the builder",
            "description": "what a dream guy",
        },
    ]


def test_cosine_similarity() -> None:
    ## Colinear vectors should have a similarity of 1
    arr1 = np.array([1, 0])
    arr2 = np.array([2, 0])

    assert cosine_similarity(arr1, arr2) == 1.0

    ## Perpendicular vectors should have a similarity of 0

    arr1 = np.array([1, 0])
    arr2 = np.array([0, 1])

    assert cosine_similarity(arr1, arr2) == 0.0

    ## Ensure normalization is properly done

    arr1 = np.array([2, 1])
    arr2 = np.array([1, 2])

    assert cosine_similarity(arr1, arr2) == pytest.approx(
        4 / 5
    )  # (2*1 + 2*1)/(sqrt(2^2+1^2)^2) = 4/5

    ## Opposite vectors should return negative 1

    arr1 = np.array([1, 0])
    arr2 = np.array([-3, 0])

    assert cosine_similarity(arr1, arr2) == -1.0


def test_generate_embedding() -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding("bla")
    assert type(embedding) is np.ndarray
    assert embedding.shape == (384,)


def test_load_or_create_embeddings() -> None:
    movies = get_sample_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)

    ## Checking the embeddings are correctly built
    assert type(semantic_search.embeddings) is np.ndarray
    assert semantic_search.embeddings.shape == (3, 384)

    ## Checking the document map is correctly built
    assert len(semantic_search.document_map) == 3
    assert semantic_search.document_map[1] == {
        "id": 1,
        "title": "Cats, cats and dogs",
        "description": "rule the earth",
    }
    assert semantic_search.document_map[2] == {
        "id": 2,
        "title": "flat earth confirmed",
        "description": "in your dreams.",
    }

    assert semantic_search.document_map[3] == {
        "id": 3,
        "title": "bob the builder",
        "description": "what a dream guy",
    }


def test_search() -> None:
    movies = get_sample_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)

    matches = semantic_search.search("a movie about planets", 1)
    assert matches[0]["title"] == "flat earth confirmed"
