from django.db import models
from accounts.models import CustomUser


class Model(models.Model):
    name = models.CharField(max_length=255, default="GPT-3.5-Turbo")
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
