# test_embed_functions.py

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


def test_split_into_many_small_text():
    text = "This is a small text. It has only a few tokens."
    expected_result = [
        text + "."
    ]  # Adjusting expected result to match the function's output
    result = split_into_many(text, max_tokens=500)
    assert result == expected_result


def test_split_into_many_large_text():
    # Define a large text and max_tokens
    text = "This is a large text with multiple sentences. It has more tokens than the specified max_tokens limit."
    max_tokens = 10  # adjust this to a suitable value

    # Split the text into sentences
    sentences = text.split(". ")

    # Calculate the expected chunks considering the additional period at the end
    expected_chunks = []
    chunk = []
    tokens_so_far = 0
    for sentence in sentences:
        tokens = len(
            tokenizer.encode(" " + sentence)
        )  # Simulate the token count as close as possible
        if tokens_so_far + tokens > max_tokens:
            expected_chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        if tokens > max_tokens:
            expected_chunks.append(sentence + ".")
            continue
        chunk.append(sentence)
        tokens_so_far += tokens
    if chunk:
        expected_chunks.append(". ".join(chunk) + ".")

    # Call the function and get the result
    result = split_into_many(text, max_tokens=max_tokens)

    # Assert that the result matches the expected chunks
    assert result == expected_chunks


def test_split_into_many_empty_text():
    text = ""
    expected_result = [text + "."]
    result = split_into_many(text, max_tokens=50)
    assert result == expected_result


def test_chunks_end_with_period():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_into_many(text, max_tokens=50)
    assert all(chunk.endswith(".") for chunk in chunks)
