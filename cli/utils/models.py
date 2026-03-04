from typing import TypedDict


class MovieMatch(TypedDict):
    title: str
    score: float
    description: str


class MovieEntry(TypedDict):
    id: int
    title: str
    description: str


class BM25Match(TypedDict):
    doc_id: int
    bm25_score: float
