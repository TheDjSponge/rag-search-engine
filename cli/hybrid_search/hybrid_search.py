import os

from cli.hybrid_search.normalisation import normalize
from cli.semantic_search.chunked_semantic_search import ChunkedSemanticSearch
from cli.tf_idf.inverted_index import InvertedIndex
from cli.utils.models import (
    BM25Match,
    HybridMatchItem,
    MovieEntry,
    RRFMatchItem,
)


class HybridSearch:
    def __init__(self, documents: list[MovieEntry]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def weighted_search(
        self, query: str, alpha: float, limit: int
    ) -> list[tuple[int, HybridMatchItem]]:
        bm25_matches = self._bm25_search(
            query, limit=limit * 500
        )  # doc_id, bm25_score
        semantic_matches = self.semantic_search.search_chunks(
            query, limit=500 * limit
        )  # id, score

        bm25_ids = []
        bm25_scores = []
        for bm25_match in bm25_matches:
            bm25_ids.append(bm25_match["doc_id"])
            bm25_scores.append(bm25_match["bm25_score"])
        normalized_bm25_scores = normalize(bm25_scores)

        semantic_ids = []
        semantic_scores = []
        for semantic_match in semantic_matches:
            semantic_ids.append(semantic_match["id"])
            semantic_scores.append(semantic_match["score"])
        normalized_semantic_scores = normalize(semantic_scores)

        combined_score: dict[int, HybridMatchItem] = {}
        for id, score in zip(
            semantic_ids, normalized_semantic_scores, strict=True
        ):
            movie_info: HybridMatchItem = {
                "title": self.documents[id - 1]["title"],
                "description": self.documents[id - 1]["description"],
                "semantic_score": score,
                "keyword_score": 0.0,
                "hybrid_score": 0.0,
            }
            combined_score[id] = movie_info

        for id, score in zip(bm25_ids, normalized_bm25_scores, strict=True):
            if id not in combined_score:
                movie_info_bm25: HybridMatchItem = {
                    "title": self.documents[id - 1]["title"],
                    "description": self.documents[id - 1]["description"],
                    "semantic_score": 0.0,
                    "keyword_score": score,
                    "hybrid_score": 0.0,
                }
                combined_score[id] = movie_info_bm25
            else:
                combined_score[id]["keyword_score"] = score

        for key in combined_score:
            movie = combined_score[key]
            combined_score[key]["hybrid_score"] = hybrid_score(
                movie["keyword_score"], movie["semantic_score"], alpha
            )
        sorted_hybrid_matches = sorted(
            combined_score.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True,
        )
        return sorted_hybrid_matches[:limit]

    def _bm25_search(self, query: str, limit: int) -> list[BM25Match]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(
        self, query: str, k: int, limit: int = 10
    ) -> list[tuple[int, RRFMatchItem]]:
        bm25_matches = self._bm25_search(
            query, limit=limit * 500
        )  # doc_id, bm25_score
        semantic_matches = self.semantic_search.search_chunks(
            query, limit=500 * limit
        )  # id, score

        bm25_ids = []
        bm25_ranks = []
        for idx, bm25_match in enumerate(bm25_matches, start=1):
            bm25_ids.append(bm25_match["doc_id"])
            bm25_ranks.append(idx)

        semantic_ids = []
        semantic_ranks = []
        for idx, semantic_match in enumerate(semantic_matches, start=1):
            semantic_ids.append(semantic_match["id"])
            semantic_ranks.append(idx)

        combined_score: dict[int, RRFMatchItem] = {}
        for id, score in zip(semantic_ids, semantic_ranks, strict=True):
            movie_info: RRFMatchItem = {
                "title": self.documents[id - 1]["title"],
                "description": self.documents[id - 1]["description"],
                "keyword_rank": None,
                "semantic_rank": score,
                "rrf_score": 0.0,
            }
            combined_score[id] = movie_info

        for id, score in zip(bm25_ids, bm25_ranks, strict=True):
            if id not in combined_score:
                movie_info_bm: RRFMatchItem = {
                    "title": self.documents[id - 1]["title"],
                    "description": self.documents[id - 1]["description"],
                    "keyword_rank": score,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }
                combined_score[id] = movie_info_bm
            else:
                combined_score[id]["keyword_rank"] = score

        for key in combined_score:
            movie = combined_score[key]
            keyword_rrf_score = (
                rrf_score(movie["keyword_rank"])
                if movie["keyword_rank"] is not None
                else 0
            )
            semantic_rrf_score = (
                rrf_score(movie["semantic_rank"])
                if movie["semantic_rank"] is not None
                else 0
            )

            combined_score[key]["rrf_score"] = (
                keyword_rrf_score + semantic_rrf_score
            )
        sorted_hybrid_matches = sorted(
            combined_score.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )
        return sorted_hybrid_matches[:limit]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score
