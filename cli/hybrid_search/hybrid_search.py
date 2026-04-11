import os

from cli.semantic_search.chunked_semantic_search import ChunkedSemanticSearch
from cli.tf_idf.inverted_index import InvertedIndex
from cli.utils.models import BM25Match, MovieEntry


class HybridSearch:
    def __init__(self, documents: list[MovieEntry]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def weighted_search(self, query: str, alpha: float, limit: int) -> None:
        # aggregated_score = {}
        bm25_matches = self._bm25_search(
            query, limit=limit * 500
        )  # doc_id, bm25_score
        # semantic_matches = self.semantic_search.search_chunks(
        #     query, limit=500 * limit
        # )  # id, score
        print(bm25_matches)

    def _bm25_search(self, query: str, limit: int) -> list[BM25Match]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query: str, k: float, limit: int = 10) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
