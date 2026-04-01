from typing import Any, TypedDict


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class MovieMatchSemantic(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


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
