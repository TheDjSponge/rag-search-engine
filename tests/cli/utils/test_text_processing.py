from cli.utils.text_processing import (
    format_string,
    remove_stopwords,
    stem_tokens,
    tokenize,
)


def test_remove_stopwords() -> None:
    stopwords_list = ["a", "an"]

    query = ["a", "cat", "ate", "an", "apple"]
    filtered_query = remove_stopwords(query, stopwords_list)

    expected = ["cat", "ate", "apple"]

    assert filtered_query == expected


def test_format_string() -> None:
    query = "This dog's house is amazing, I love it!"
    formatted_query = format_string(query)

    expected = "this dogs house is amazing i love it"

    assert formatted_query == expected


def test_tokenize() -> None:
    query = " I should be tokenized    "
    tokenized_query = tokenize(query)

    expected = ["I", "should", "be", "tokenized"]

    assert tokenized_query == expected


def test_stemming() -> None:
    querry = ["cat", "running", "tree", "falls"]
    stemmed_querry = stem_tokens(querry)

    expected_result = ["cat", "run", "tree", "fall"]

    assert stemmed_querry == expected_result
