import os

import numpy as np
from sentence_transformers import SentenceTransformer

from cli.utils.constants import CACHE_DIR, EMBEDDING_MODEL
from cli.utils.files import load_movies
from cli.utils.models import MovieEntry, MovieMatch


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text: str) -> None:

    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:

    semantic_search = SemanticSearch()
    documents = load_movies("./data/movies.json")["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(
    vec1: np.ndarray | list[float], vec2: np.ndarray | list[float]
) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings: np.ndarray | None = None
        self.documents: list[MovieEntry] | None = None
        self.document_map: dict[int, MovieEntry] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def search(self, query: str, limit: int) -> list[MovieMatch]:
        embedded_query = self.generate_embedding(query)
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "Embeddings are not loaded or built, please run verify_embeddings before searching"
            )

        scores = []
        for id, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(embedded_query, doc_embedding)
            scores.append((self.documents[id], similarity))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        selected: list[MovieMatch] = []
        for score in sorted_scores:
            selected.append(
                {
                    "score": score[1],
                    "title": score[0]["title"],
                    "description": score[0]["description"],
                }
            )
            if len(selected) >= limit:
                break
        return selected

    def load_or_create_embeddings(
        self, documents: list[MovieEntry]
    ) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if (self.embeddings is not None) and (
                len(self.embeddings) == len(self.documents)
            ):
                return self.embeddings
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings

    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text.replace(" ", "")) == 0:
            ValueError(
                "The provided text is empty or contains whitespaces only"
            )
        embeddings = self.model.encode(text)
        return np.array(embeddings)

    def build_embeddings(self, documents: list[MovieEntry]) -> np.ndarray:
        self.documents = documents
        document_representations = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_representations.append(
                f"{document['title']}: {document['description']}"
            )
        self.embeddings = self.model.encode(
            document_representations, show_progress_bar=True
        )
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
