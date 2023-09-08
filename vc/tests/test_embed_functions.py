import pytest
import pandas as pd
from vc.embed_functions import remove_newlines, split_into_many
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

# test remove_newlines


def test_remove_newlines_replaces_newline_characters():
    test_series = pd.Series(["This is line 1\nThis is line 2", "Another line\nAnother"])
    result = remove_newlines(test_series)
    expected_series = pd.Series(
        ["This is line 1 This is line 2", "Another line Another"]
    )
    assert result.equals(expected_series)


def test_remove_newlines_replaces_escaped_newlines():
    test_series = pd.Series(
        ["This is line 1\\nThis is line 2", "Another line\\nAnother"]
    )
    result = remove_newlines(test_series)
    expected_series = pd.Series(
        ["This is line 1 This is line 2", "Another line Another"]
    )
    assert result.equals(expected_series)


def test_remove_newlines_replaces_double_spaces():
    test_series = pd.Series(["Double  space", "Another  double"])
    result = remove_newlines(test_series)
    expected_series = pd.Series(["Double space", "Another double"])
    assert result.equals(expected_series)


def test_remove_newlines_handles_empty_strings():
    test_series = pd.Series(["", "\n", "\\n", "  "])
    result = remove_newlines(test_series)
    expected_series = pd.Series(["", " ", " ", " "])
    assert result.equals(expected_series)


def test_remove_newlines_preserves_existing_single_spaces():
    test_series = pd.Series(["No  changes", "Should  remain"])
    result = remove_newlines(test_series)
    expected_series = pd.Series(["No changes", "Should remain"])
    assert result.equals(expected_series)


# test split_into_many


def count_tokens(text):
    return sum(1 for _ in tokenizer(text))


# def test_splits_text_into_chunks_less_than_max_tokens():
#     text = "This is sentence one. This is sentence two. This is sentence three."
#     chunks = split_into_many(
#         text, max_tokens=20
#     )  # set max_tokens to a small number for testing
#     assert all(count_tokens(chunk) <= 20 for chunk in chunks)


# def test_skips_sentences_exceeding_max_tokens():
#     text = "This sentence is short. This sentence is deliberately designed to exceed the max token limit."
#     chunks = split_into_many(text, max_tokens=10)
#     assert (
#         "This sentence is deliberately designed to exceed the max token limit."
#         not in chunks
#     )


# def test_handles_empty_text():
#     text = ""
#     chunks = split_into_many(text, max_tokens=50)
#     assert chunks == []


# def test_handles_single_sentence():
#     text = "This is a single sentence."
#     chunks = split_into_many(text, max_tokens=50)
#     assert chunks == ["This is a single sentence."]


def test_chunks_end_with_period():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_into_many(text, max_tokens=50)
    assert all(chunk.endswith(".") for chunk in chunks)
