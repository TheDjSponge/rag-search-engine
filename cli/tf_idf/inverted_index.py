import os
import math
import pickle
from typing import List, Dict
from collections import Counter
from cli.utils.text_processing import prepare_and_tokenize
from cli.utils.constants import CACHE_DIR, BM25_K1, BM25_B, RETRIEVAL_LIMIT


class InvertedIndex:
    def __init__(self) -> None:
        # Index is the mapping between token ID -> documents. Structured as "hello" : [0,10,20]
        # Where number are supposed to be document IDs
        self.index = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")

        # Docmaps is the relation between document IDs (0,10,20) to the full document object {title + description}
        self.docmap = {}
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

        # Term frequency holds for each document ID a counter of the words present in it.
        self.term_frequencies = {}
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

        # Stores length of each doc
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self, movies: List[Dict]) -> None:
        print("~ Building inverse index, document map and counting term frequencies ~")
        for movie in movies:
            movie_text = f"{movie["title"]} {movie["description"]}"
            movie_id = movie["id"]
            self.__add_document(movie_id, movie_text)
            self.docmap[movie_id] = movie_text

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_term = prepare_and_tokenize(term)
        if len(tokenized_term) != 1:
            raise ValueError("term should contain a single token")
        token = tokenized_term[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokenized_term = prepare_and_tokenize(term)
        if len(tokenized_term) != 1:
            raise ValueError("term should contain a single token")
        token = tokenized_term[0]

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_document(token))

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = prepare_and_tokenize(term)
        if len(tokenized_term) != 1:
            raise ValueError("term should contain a single token")
        token = tokenized_term[0]

        total_docs = len(self.docmap)
        document_frequency = len(self.index[token])
        return math.log(
            (total_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ):
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf

    def bm25(self, doc_id: int, term: str):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = RETRIEVAL_LIMIT):
        tokenized_query = prepare_and_tokenize(query)
        scores = {}
        for token in tokenized_query:
            for doc_id in self.index[token]:
                if doc_id in scores:
                    scores[doc_id] += self.bm25(doc_id=doc_id, term=token)
                else:
                    scores[doc_id] = self.bm25(doc_id=doc_id, term=token)

        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        matches = []
        for doc_id, score in sorted_scores.items():
            matches.append(dict(doc_id=doc_id, bm25_score=score))
            if len(matches) >= limit:
                break
        return matches

    def get_document(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), []))

    def save(self) -> None:
        print(
            "~ Saving index.pkl, docmap.pkl and term_frequencies.pkl files under ./cache ~"
        )
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        with open(self.index_path, "wb") as index:
            pickle.dump(self.index, index)

        with open(self.docmap_path, "wb") as docmap:
            pickle.dump(self.docmap, docmap)

        with open(self.term_frequencies_path, "wb") as tf:
            pickle.dump(self.term_frequencies, tf)

        with open(self.doc_lengths_path, "wb") as doc_lengths:
            pickle.dump(self.doc_lengths, doc_lengths)

    def load(self) -> None:
        if (not os.path.exists("./cache/index.pkl")) or (
            not os.path.exists("./cache/docmap.pkl")
        ):
            raise FileNotFoundError("index.pkl or docmap.pkl not found under ./cache")

        with open(self.index_path, "rb") as index:
            self.index = pickle.load(index)

        with open(self.docmap_path, "rb") as docmap:
            self.docmap = pickle.load(docmap)

        with open(self.term_frequencies_path, "rb") as tf:
            self.term_frequencies = pickle.load(tf)

        with open(self.doc_lengths_path, "rb") as doc_lengths:
            self.doc_lengths = pickle.load(doc_lengths)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = prepare_and_tokenize(text)
        for token in set(tokenized_text):
            if token in self.index:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]
        self.term_frequencies[doc_id] = Counter(tokenized_text)
        self.doc_lengths[doc_id] = len(tokenized_text)

    def __get_avg_doc_length(self) -> float:
        doc_token_counts = list(self.doc_lengths.values())
        if len(doc_token_counts) == 0:
            return 0.0
        else:
            return sum(doc_token_counts) / len(doc_token_counts)
