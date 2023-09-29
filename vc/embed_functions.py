# embed_functions.py
import re
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def split_into_many(text, max_tokens=500, preserve_sentences=False):
    sentences = text.split(". ")

    chunks = []
    chunk = ""
    tokens_so_far = 0

    # Loop through the sentences
    for sentence in sentences:
        tokens = len(tokenizer.encode(" " + sentence))

        if preserve_sentences:
            # If the sentence itself is longer than max_tokens, append it directly to chunks
            if tokens > max_tokens:
                # If there's any current chunk, append it to chunks before appending the long sentence
                if chunk:
                    chunks.append(chunk.strip())
                    chunk = ""
                    tokens_so_far = 0
                chunks.append(sentence.strip())
                continue
            # If adding the current sentence to the chunk exceeds max_tokens, start a new chunk
            elif tokens_so_far + tokens > max_tokens:
                chunks.append(chunk.strip())
                chunk = ""
                tokens_so_far = 0
            chunk += sentence + ". "
            tokens_so_far += tokens
        else:
            # Existing logic for when preserve_sentences is False
            if tokens > max_tokens:
                words = sentence.split()
                sub_chunk = ""
                for word in words:
                    word_tokens = len(tokenizer.encode(" " + word))
                    if tokens_so_far + word_tokens > max_tokens:
                        chunks.append(sub_chunk.strip())
                        sub_chunk = ""
                        tokens_so_far = 0
                    sub_chunk += word + " "
                    tokens_so_far += word_tokens
                chunks.append(sub_chunk.strip())
                tokens_so_far = 0
                continue

            if tokens_so_far + tokens > max_tokens:
                chunks.append(chunk.strip())
                chunk = ""
                tokens_so_far = 0

            chunk += sentence + ". "
            tokens_so_far += tokens

    # If there is any remaining content in the chunk, append it to chunks.
    if chunk:
        chunks.append(chunk.strip())

    return chunks
