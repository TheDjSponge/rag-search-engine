from typing import Any, NotRequired, TypedDict


class RRFMatchItem(TypedDict):
    title: str
    description: str
    semantic_rank: int | None
    keyword_rank: int | None
    rrf_score: float
    rerank_score: NotRequired[float]
    cross_encoder_score: NotRequired[float]


class HybridMatchItem(TypedDict):
    title: str
    description: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float


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
