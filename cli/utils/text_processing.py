import math
import re
import string

from nltk.stem import PorterStemmer

from .files import load_stopwords


def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str]:
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def format_string(to_format: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    lower_string = to_format.lower()
    translated_string = lower_string.translate(translation_table)
    return translated_string


def tokenize(to_tokenize: str) -> list[str]:
    return to_tokenize.split()


def prepare_and_tokenize(text: str) -> list[str]:
    stopwords_list = load_stopwords("./data/stopwords.txt")
    formatted_text = format_string(text)
    tokenized_text = tokenize(formatted_text)
    cleaned_text = remove_stopwords(tokenized_text, stopwords_list)
    stemmed_tokens = stem_tokens(cleaned_text)
    return stemmed_tokens


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    if text == "":
        raise ValueError("text to chunk can't be empty")

    splitted_words = text.split(" ")
    num_chunks = math.ceil(len(splitted_words) // chunk_size)
    chuncked_text = []
    for chunk in range(num_chunks):
        chuncked_text.append(
            " ".join(
                splitted_words[chunk * chunk_size : (chunk + 1) * chunk_size]
            )
        )
    remains = " ".join(splitted_words[(num_chunks) * chunk_size :])

    if remains != "":
        chuncked_text.append(remains)

    return chuncked_text


def chunk_text_with_overlap(
    text: str, chunk_size: int = 200, overlap: int = 0
) -> list[str]:
    if text == "":
        raise ValueError("text to chunk can't be empty")

    splitted_words = text.split(" ")
    step = chunk_size - overlap
    ptr = 0
    chunks = []
    while ptr <= (len(splitted_words)):
        chunk = splitted_words[ptr : ptr + chunk_size]
        chunks.append(" ".join(chunk))
        if ptr + chunk_size >= len(splitted_words):
            break
        ptr += step
    return chunks


def chunk_sentences_semantic(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    text = text.strip(" ")
    if text == "":
        return []
    splitted_words = re.split(r"(?<=[.!?])\s+", text)
    step = max_chunk_size - overlap
    ptr = 0
    chunks = []
    while ptr <= (len(splitted_words)):
        chunk = splitted_words[ptr : ptr + max_chunk_size]
        if chunk == []:
            break

        for id, sentence in enumerate(chunk):
            chunk[id] = sentence.strip(" ")
        chunks.append(" ".join(chunk))
        if ptr + max_chunk_size >= len(splitted_words):
            break
        ptr += step
    return chunks
