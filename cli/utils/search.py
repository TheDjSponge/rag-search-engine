import string
from typing import Dict, List
from .files import load_stopwords
from nltk.stem import PorterStemmer


def find_matching_movies(movies: Dict[str, List[Dict]], query: str) -> List[str]:
    matched = []
    movies_list = movies["movies"]
    stopwords_list = load_stopwords("./data/stopwords.txt")

    query = remove_stopwords(query, stopwords_list)
    tokenized_query = tokenize(format_string(query))
    stemmed_query = stem_tokens(tokenized_query)

    for movie in movies_list:
        movie_title = remove_stopwords(movie["title"], stopwords_list)
        tokenized_title = tokenize(format_string(movie_title))
        stemmed_title = stem_tokens(tokenized_title)

        if has_token_intersection(stemmed_query, stemmed_title):
            matched.append(movie["title"])

    return matched


def print_matched_movies(matched_list: List[str]) -> None:
    print("\n--- Words found through direct matching ---\n")
    for movie_title in matched_list:
        print(f"- {movie_title}")


def format_string(to_format: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    lower_string = to_format.lower()
    translated_string = lower_string.translate(translation_table)
    return translated_string


def tokenize(to_tokenize: str) -> List[str]:
    return to_tokenize.split()


def has_token_intersection(tokens_1: List[str], tokens_2: List[str]):
    return len(set(tokens_1).intersection(set(tokens_2))) > 0


def remove_stopwords(text: str, stopwords: List[str]):
    tokenized_text = tokenize(text)
    for stopword in stopwords:
        if stopword in tokenized_text:
            tokenized_text.remove(stopword)
    return " ".join(tokenized_text)


def stem_tokens(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
