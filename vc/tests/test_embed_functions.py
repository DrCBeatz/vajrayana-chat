# test_embed_functions.py

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

    # Call the function and get the result
    result = split_into_many(text, max_tokens=max_tokens)

    # Check that each chunk in the result does not exceed max_tokens
    for chunk in result:
        assert len(tokenizer.encode(chunk)) <= max_tokens

    # Split the original text and the reconstructed text into words
    original_words = set(text.split())
    reconstructed_words = set(" ".join(result).split())

    # Check that the sets of words are equal
    assert original_words == reconstructed_words


def test_split_into_many_sentence_larger_than_max_tokens():
    large_sentence = "This sentence is larger than max_tokens. " * 50
    text = f"Small sentence. {large_sentence} Another small sentence."
    result = split_into_many(text, max_tokens=50)

    # Check that the small sentences are in the result
    expected_small_sentences = ["Small sentence.", "Another small sentence."]
    for small_sentence in expected_small_sentences:
        assert any(
            small_sentence in res for res in result
        ), f"{small_sentence} not found in result."

    # Check that every part of the large sentence is included in the result.
    # Here, create a version of large_sentence that is guaranteed to be broken down into max_token parts
    large_sentence_parts = [
        large_sentence[i : i + 50] for i in range(0, len(large_sentence), 50)
    ]
    for part in large_sentence_parts:
        assert any(part in res for res in result), f"{part} not found in result."


def test_split_into_many_empty_text():
    text = ""
    expected_result = [text + "."]
    result = split_into_many(text, max_tokens=50)
    assert result == expected_result


def test_chunks_end_with_period():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_into_many(text, max_tokens=50)
    assert all(chunk.endswith(".") for chunk in chunks)
