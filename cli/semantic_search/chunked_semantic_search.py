import json
import os

import numpy as np

from cli.semantic_search.semantic_search import (
    SemanticSearch,
    cosine_similarity,
)
from cli.utils.constants import CACHE_DIR, SCORE_PRECISION
from cli.utils.models import ChunkScore, MovieEntry, MovieMatchSemantic
from cli.utils.text_processing import chunk_sentences_semantic


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict[str, int]] | None = None
        self.chunk_embeddings_path = os.path.join(
            CACHE_DIR, "chunk_embeddings.npy"
        )
        self.chunk_metadata_path = os.path.join(
            CACHE_DIR, "chunk_metadata.json"
        )

    def build_chunk_embeddings(self, documents: list[MovieEntry]) -> np.ndarray:
        self.documents = documents
        all_chunks = []
        chunk_metadata = []
        for document in documents:
            if document["description"] == "":
                continue
            document_chunks = chunk_sentences_semantic(
                document["description"], max_chunk_size=4, overlap=1
            )
            nb_chunks = len(document_chunks)
            for id, chunk in enumerate(document_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": document["id"],
                        "chunk_idx": id,
                        "total_chunks": nb_chunks,
                    }
                )
            self.document_map[document["id"]] = document
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as file:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self, documents: list[MovieEntry]
    ) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path) as file:
                self.chunk_metadata = json.load(file)["chunks"]
            if self.chunk_embeddings is not None:
                return self.chunk_embeddings
        self.chunk_embeddings = self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunks(
        self, query: str, limit: int = 10
    ) -> list[MovieMatchSemantic]:
        embedded_query = self.generate_embedding(query)
        if self.chunk_embeddings is None or self.documents is None:
            raise ValueError(
                "Embeddings are not loaded or built, please run embed_chunks before searching"
            )

        chunk_scores: list[ChunkScore] = []
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("no chunk embeddings found to search in.")

        for chunk_id, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk, embedded_query)
            chunk_scores.append(
                {
                    "chunk_idx": chunk_id,
                    "movie_idx": self.chunk_metadata[chunk_id]["movie_idx"],
                    "score": score,
                }
            )
        movie_to_score_map: dict[int, float] = {}
        for chunk_score in chunk_scores:
            if (chunk_score["movie_idx"] not in movie_to_score_map) or (
                chunk_score["score"]
                > movie_to_score_map[chunk_score["movie_idx"]]
            ):
                movie_to_score_map[chunk_score["movie_idx"]] = chunk_score[
                    "score"
                ]

        sorted_scores = sorted(
            movie_to_score_map.items(), key=lambda x: x[1], reverse=True
        )

        selected: list[MovieMatchSemantic] = []
        for movie_score in sorted_scores:
            movie_idx = movie_score[0] - 1  # List index

            selected.append(
                {
                    "id": movie_idx + 1,  # Dataset index
                    "title": self.documents[movie_idx]["title"],
                    "document": self.documents[movie_idx]["description"][:100],
                    "score": round(movie_score[1], SCORE_PRECISION),
                    "metadata": (
                        self.chunk_metadata[movie_idx]
                        if self.chunk_metadata is not None
                        else {}
                    ),
                }
            )
            if len(selected) >= limit:
                break
        return selected
