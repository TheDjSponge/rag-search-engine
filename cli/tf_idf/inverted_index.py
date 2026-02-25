import os
import math
import pickle
from typing import List, Dict
from collections import Counter
from cli.utils.text_processing import prepare_and_tokenize


class InvertedIndex:
    def __init__(self) -> None:
        # Index is the mapping between token ID -> documents. Structured as "hello" : [0,10,20]
        # Where number are supposed to be document IDs
        self.index = {}

        # Docmaps is the relation between document IDs (0,10,20) to the full document object {title + description}
        self.docmap = {}

        # Term frequency holds for each document ID a counter of the words present in it.
        self.term_frequencies = {}

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
        document_frequency = len(self.get_document(token))
        print(document_frequency)
        return math.log(
            (total_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1
        )

    def get_document(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), []))

    def save(self) -> None:
        print(
            "~ Saving index.pkl, docmap.pkl and term_frequencies.pkl files under ./cache ~"
        )
        if not os.path.exists("./cache"):
            os.makedirs("./cache")

        with open("./cache/index.pkl", "wb") as index:
            pickle.dump(self.index, index)

        with open("./cache/docmap.pkl", "wb") as docmap:
            pickle.dump(self.docmap, docmap)

        with open("./cache/term_frequencies.pkl", "wb") as tf:
            pickle.dump(self.term_frequencies, tf)

    def load(self) -> None:
        if (not os.path.exists("./cache/index.pkl")) or (
            not os.path.exists("./cache/docmap.pkl")
        ):
            raise FileNotFoundError("index.pkl or docmap.pkl not found under ./cache")

        with open("./cache/index.pkl", "rb") as index:
            self.index = pickle.load(index)

        with open("./cache/docmap.pkl", "rb") as docmap:
            self.docmap = pickle.load(docmap)

        with open("./cache/term_frequencies.pkl", "rb") as tf:
            self.term_frequencies = pickle.load(tf)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = prepare_and_tokenize(text)
        for token in set(tokenized_text):
            if token in self.index:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]
        self.term_frequencies[doc_id] = Counter(tokenized_text)
