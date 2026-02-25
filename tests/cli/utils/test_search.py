from cli.utils.search import (
    remove_stopwords,
    format_string,
    tokenize,
    stem_tokens,
    has_token_intersection,
)


def test_remove_stopwords():
    stopwords_list = ["a", "an"]

    query = "a cat ate an apple"
    filtered_query = remove_stopwords(query, stopwords_list)

    expected = "cat ate apple"

    assert filtered_query == expected


def test_format_string():
    query = "This dog's house is amazing, I love it!"
    formatted_query = format_string(query)

    expected = "this dogs house is amazing i love it"

    assert formatted_query == expected


def test_tokenize():
    query = " I should be tokenized    "
    tokenized_query = tokenize(query)

    expected = ["I", "should", "be", "tokenized"]

    assert tokenized_query == expected


def test_stemming():
    querry = ["cat", "running", "tree", "falls"]
    stemmed_querry = stem_tokens(querry)

    expected_result = ["cat", "run", "tree", "fall"]

    assert stemmed_querry == expected_result


def test_token_intersection():
    set1 = ["yes", "no"]
    set2 = ["no", "maybe"]
    set3 = ["lol"]
    assert has_token_intersection(set1, set2)
    assert not has_token_intersection(set1, set3)
