import pytest

from cli.utils.text_processing import (
    chunk_sentences_semantic,
    chunk_text,
    chunk_text_with_overlap,
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


def test_chunk_text() -> None:
    to_chunck = "oh my god I love dogs so much it's crazy"
    expected = ["oh my god", "I love dogs", "so much it's", "crazy"]

    chuncked = chunk_text(to_chunck, chunk_size=3)

    assert chuncked == expected

    to_chunck = "Simple working example"
    expected = ["Simple", "working", "example"]

    chuncked = chunk_text(to_chunck, chunk_size=1)

    assert chuncked == expected


@pytest.mark.parametrize(
    "text, chunk_size, overlap, expected",
    [
        # no overlap: clean equal-sized chunks
        ("a b c d e f", 2, 0, ["a b", "c d", "e f"]),
        # with overlap: step=2, windows at 0, 2, 4
        ("a b c d e f g", 4, 2, ["a b c d", "c d e f", "e f g"]),
        # last chunk smaller than chunk_size
        ("a b c d e", 3, 0, ["a b c", "d e"]),
        # text shorter than chunk_size: single chunk
        ("hello world", 10, 0, ["hello world"]),
        # single-word chunks
        ("x y z", 1, 0, ["x", "y", "z"]),
    ],
)
def test_chunk_text_with_overlap(
    text: str, chunk_size: int, overlap: int, expected: list[str]
) -> None:
    assert (
        chunk_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
        == expected
    )


@pytest.mark.parametrize(
    "text, max_chunk_size, overlap, expected",
    [
        # no overlap: clean equal-sized chunks
        ("a. b. c. d. e. f.", 2, 0, ["a. b.", "c. d.", "e. f."]),
        # with overlap: step=2, windows at 0, 2, 4
        (
            "a. b. c. d. e. f. g.",
            4,
            2,
            ["a. b. c. d.", "c. d. e. f.", "e. f. g."],
        ),
        # last chunk smaller than chunk_size
        ("a. b. c. d! e?", 3, 0, ["a. b. c.", "d! e?"]),
        # text shorter than chunk_size: single chunk
        ("hello world?", 10, 0, ["hello world?"]),
        # single-word chunks
        ("x. y? z!", 1, 0, ["x.", "y?", "z!"]),
        # leading and trailing spaces
        ("   abc.   xyz.", 1, 0, ["abc.", "xyz."]),
        # text without punctuation
        ("abc", 2, 0, ["abc"]),
        # only whitespaces
        ("   ", 1, 0, []),
        # empty string
        ("", 1, 0, []),
    ],
)
def test_chunk_sentences_semantic(
    text: str, max_chunk_size: int, overlap: int, expected: list[str]
) -> None:
    assert (
        chunk_sentences_semantic(
            text, max_chunk_size=max_chunk_size, overlap=overlap
        )
        == expected
    )
