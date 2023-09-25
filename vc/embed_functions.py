# embed_functions.py
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def split_into_many(text, max_tokens=500):
    # Split the text into sentences
    sentences = text.split(". ")

    chunks = []
    chunk = ""
    tokens_so_far = 0

    # Loop through the sentences
    for sentence in sentences:
        tokens = len(tokenizer.encode(" " + sentence))

        # If the sentence itself has more tokens than max_tokens,
        # split it further into smaller chunks
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

        # If adding the next sentence to the current chunk exceeds max_tokens,
        # add the current chunk to chunks and start a new chunk
        if tokens_so_far + tokens > max_tokens:
            chunks.append(chunk.strip())
            chunk = ""
            tokens_so_far = 0

        chunk += sentence + ". "
        tokens_so_far += tokens

    if chunk:
        chunks.append(chunk.strip())

    return chunks
