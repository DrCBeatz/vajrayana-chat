from django.db import models
from accounts.models import CustomUser
import pandas as pd
import openai
from django.core.files import File
import tiktoken
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from .embed_functions import remove_newlines, split_into_many

from youtube_transcript_api import YouTubeTranscriptApi

tokenizer = tiktoken.get_encoding("cl100k_base")


def clean_text(text):
    # Replace multiple newline characters with a single newline
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    # Remove any remaining whitespace
    text = text.strip()
    return text


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
    html_url = models.URLField(max_length=2000, blank=True, null=True)
    youtube_url = models.URLField(max_length=2000, blank=True, null=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        content_changed = False

        # Check if the instance is new or the content has changed
        if not self.pk:
            content_changed = bool(self.content)
        else:
            original = Document.objects.get(pk=self.pk)
            content_changed = original.content != self.content

        # If content hasn't changed, check for html_url
        if not content_changed and self.html_url and not self.content:
            response = requests.get(self.html_url)
            soup = BeautifulSoup(response.content, "html.parser")
            # Extract text from the HTML
            raw_text = soup.get_text()
            # Clean the text
            self.content = clean_text(raw_text)
            content_changed = True

        # If content hasn't changed, check for document
        if not content_changed and self.document and not self.content:
            # Check if the uploaded file is a PDF
            if self.document.name.endswith(".pdf"):
                # Extract text from PDF using PdfReader
                pdf_reader = PdfReader(self.document)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                self.content = text_content
            else:
                # For text files, read and decode
                file_content = self.document.read()
                self.content = file_content.decode("utf-8")
            content_changed = True

        if not content_changed and self.youtube_url and not self.content:
            video_id = self.youtube_url.split("v=")[1].split("&")[
                0
            ]  # extract video ID from URL
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry["text"] for entry in transcript])
                self.content = transcript_text
                content_changed = True
            except Exception as e:
                # Handle errors (e.g., video doesn't have a transcript, wrong video ID, etc.)
                print(f"Error fetching transcript: {e}")

        # If content changed or is not empty on creation, generate embeddings
        if content_changed:
            self.embed()

        # Save the model instance to the database in all cases
        super().save(*args, **kwargs)

    def embed(self):
        print("running embed function")
        df = self._prepare_dataframe_from_content()
        df = self._tokenize_and_shorten_texts(df)
        self._generate_and_save_embeddings(df)

    def _prepare_dataframe_from_content(self):
        """Prepare initial dataframe from document content."""
        df = pd.DataFrame([self.content], columns=["text"])
        df["text"] = self.title + ". " + remove_newlines(df.text)
        return df

    def _tokenize_and_shorten_texts(self, df):
        """Tokenize the text and split it into chunks if needed."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        shortened = []
        for row in df.iterrows():
            if row[1]["text"] is None:
                continue
            if row[1]["n_tokens"] > 500:
                shortened += split_into_many(row[1]["text"])
            else:
                shortened.append(row[1]["text"])

        df = pd.DataFrame(shortened, columns=["text"])
        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        return df

    def _generate_and_save_embeddings(self, df):
        """Generate embeddings for the document and save them to a file."""
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

        with open(file_path, "rb") as f:
            self.embeddings.save(file_path, File(f), save=False)
