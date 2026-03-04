from sentence_transformers import SentenceTransformer
from cli.utils.constants import EMBEDDING_MODEL, CACHE_DIR
from torch import Tensor
import numpy as np
from typing import List, Dict
import os
from cli.utils.files import load_movies


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


def verify_embeddings():

    semantic_search = SemanticSearch()
    documents = load_movies("./data/movies.json")["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def search(self, query: str, limit: int) -> List:
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

        selected = []
        for score in sorted_scores:
            selected.append(
                dict(
                    score=score[1],
                    title=score[0]["title"],
                    description=score[0]["description"],
                )
            )
            if len(selected) >= limit:
                break
        return selected

    def load_or_create_embeddings(self, documents: List[Dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings

    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text.replace(" ", "")) == 0:
            ValueError("The provided text is empty or contains whitespaces only")
        embeddings = self.model.encode(text)
        return np.array(embeddings)

    def build_embeddings(self, documents: List[Dict]) -> np.ndarray:
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
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
