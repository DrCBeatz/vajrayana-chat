from django.db import models
from accounts.models import CustomUser
import pandas as pd
import openai
from django.core.files import File
import tiktoken
from .embed_functions import remove_newlines, split_into_many

tokenizer = tiktoken.get_encoding("cl100k_base")


class Model(models.Model):
    name = models.CharField(max_length=255, default="gpt-3.5-turbo")
    context_length = models.IntegerField(default=4096)
    input_token_cost = models.FloatField(default=0.0015)
    output_token_cost = models.FloatField(default=0.002)

    def __str__(self):
        return self.name


class Expert(models.Model):
    name = models.CharField(max_length=255, default="Thrangu Rinpoche")
    prompt = models.TextField(default="Answer the question based on the context below.")
    role = models.TextField(
        default="prominent tulku (reincarnate lama) in the Kagyu school of Tibetan Buddhism."
    )
    model = models.ForeignKey(Model, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class Conversation(models.Model):
    title = models.CharField(max_length=255)
    expert = models.ForeignKey(Expert, on_delete=models.CASCADE)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

    def __str__(self):
        return self.title


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    context = models.TextField()

    def __str__(self):
        return self.question


class Document(models.Model):
    title = models.CharField(max_length=255)
    expert = models.ForeignKey(Expert, on_delete=models.CASCADE)
    content = models.TextField(null=True, blank=True)
    document = models.FileField(upload_to="documents/", null=True, blank=True)
    embeddings = models.FileField(upload_to="embeddings/", null=True, blank=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        content_changed = False

        # If instance is new and content is not empty, set content_changed to True
        if not self.pk and self.content:
            content_changed = True
        # If instance is not new and content has changed, set content_changed to True
        elif self.pk:
            original = Document.objects.get(pk=self.pk)
            content_changed = original.content != self.content

        # If content changed or is not empty on creation, generate embeddings
        if content_changed:
            self.embed()

        super().save(*args, **kwargs)

    def embed(self):
        print("running embed function")
        df = pd.DataFrame([self.content], columns=["text"])
        print(df.head())
        # Set the text column to be the raw text with the newlines removed
        df["text"] = self.title + ". " + remove_newlines(df.text)

        # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and save the number of tokens to a new column
        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        # ---

        shortened = []
        count = 0
        # Loop through the dataframe
        for row in df.iterrows():
            # If the text is None, go to the next row
            if row[1]["text"] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]["n_tokens"] > 500:
                shortened += split_into_many(row[1]["text"])

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append(row[1]["text"])
            count += 1

        df = pd.DataFrame(shortened, columns=["text"])
        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        # ====
        embedding_list = []

        for index, row in df.iterrows():
            embedding = openai.Embedding.create(
                input=row["text"], engine="text-embedding-ada-002"
            )["data"][0]["embedding"]
            embedding_list.append(embedding)
            print(f"Embedding {index} of {len(df)}")

        df["embeddings"] = embedding_list

        file_path = f"embeddings/{self.title}_embeddings.parquet"
        df.to_parquet(file_path, engine="pyarrow")

        # Reference the saved parquet file in the embeddings field

        with open(file_path, "rb") as f:
            self.embeddings.save(file_path, File(f), save=False)
