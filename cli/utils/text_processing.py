import string
from .files import load_stopwords
from typing import List
from nltk.stem import PorterStemmer


def remove_stopwords(text: str, stopwords: List[str]):
    tokenized_text = tokenize(text)
    for stopword in stopwords:
        if stopword in tokenized_text:
            tokenized_text.remove(stopword)
    return " ".join(tokenized_text)


def stem_tokens(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def format_string(to_format: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    lower_string = to_format.lower()
    translated_string = lower_string.translate(translation_table)
    return translated_string


def tokenize(to_tokenize: str) -> List[str]:
    return to_tokenize.split()


def prepare_and_tokenize(text: str) -> List[str]:
    stopwords_list = load_stopwords("./data/stopwords.txt")
    cleaned_text = remove_stopwords(text, stopwords_list)
    tokenized_text = tokenize(format_string(cleaned_text))
    stemmed_tokens = stem_tokens(tokenized_text)
    return stemmed_tokens
