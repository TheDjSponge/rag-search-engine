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
