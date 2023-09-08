import pytest
from django.contrib.auth import get_user_model
from django.core.files import File
from vc.models import Model, Expert, Conversation, Message, Document
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas
from unittest import mock


def generate_pdf_content(content="this is a test pdf document"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(100, 100))
    c.drawString(10, 10, content)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


@pytest.fixture
def user(db):
    User = get_user_model()
    return User.objects.create_user("testuser", "test@example.com", "testpassword")


@pytest.fixture
def client_with_user(user, client):
    client.login(username="testuser", password="testpassword")
    return client, user


@pytest.fixture
def model_obj(db):
    return Model.objects.create(name="gpt-3.5-turbo")


@pytest.fixture
def expert_obj(db, model_obj):
    print(type(model_obj))
    return Expert.objects.create(
        name="Expert1", prompt="Prompt1", role="Role1", model=model_obj
    )


@pytest.fixture
def experts(db):
    model = Model.objects.create(name="gpt-3.5-turbo")

    return [
        Expert.objects.create(
            name="Expert1", prompt="Prompt1", role="Role1", model=model
        ),
        Expert.objects.create(
            name="Expert2", prompt="Prompt2", role="Role2", model=model
        ),
    ]


@pytest.fixture
def conversations(db, user, experts):
    return [
        Conversation.objects.create(
            title="test conversation 1", user=user, expert=experts[0]
        ),
        Conversation.objects.create(
            title="test conversation 2", user=user, expert=experts[0]
        ),
    ]


@pytest.fixture
def messages(db, conversations):
    conversation = conversations[0]
    return [
        Message.objects.create(
            conversation=conversation,
            question="Test question 1",
            answer="Test answer 1",
            context="Test context 1",
        ),
        Message.objects.create(
            conversation=conversation,
            question="Test question 2",
            answer="Test answer 2",
            context="Test context 2",
        ),
    ]


@pytest.fixture
def document_obj(db, expert_obj):
    return Document.objects.create(
        title="Document1",
        expert=expert_obj,
        content="This is a test document",
    )


@pytest.fixture
def documents(db, user, experts):
    return [
        Document.objects.create(
            expert=experts[0],
            title="Test document 1",
            embeddings="embeddings/thrangu_rinpoche_embeddings.parquet",
        ),
        Document.objects.create(
            expert=experts[1],
            title="Test document 2",
            embeddings="embeddings/mingyur_rinpoche_embeddings.parquet",
        ),
    ]


@pytest.fixture
def documents2(db, experts):
    expert1 = Expert.objects.get(name="Expert1")
    expert2 = Expert.objects.get(name="Expert2")

    # Create mock documents and save in a format (e.g., Parquet)
    # that generate_embeddings_for_experts expects.
    df1 = pd.DataFrame({"text": ["A"], "embedding": [[0.1, 0.2]]})
    df2 = pd.DataFrame({"text": ["B"], "embedding": [[0.3, 0.4]]})

    # Replace with your own logic for saving DataFrame to a file.
    df1.to_parquet("doc1.parquet")
    df2.to_parquet("doc2.parquet")

    Document.objects.create(
        title="Doc1", expert=expert1, embeddings=File(open("doc1.parquet", "rb"))
    )
    Document.objects.create(
        title="Doc2", expert=expert2, embeddings=File(open("doc2.parquet", "rb"))
    )
    return Document.objects.all()


@pytest.fixture
def mock_openai_embedding_create():
    with mock.patch("vc.views.openai.Embedding.create") as mock_create:
        mock_create.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        yield mock_create


@pytest.fixture
def mock_distances_from_embeddings():
    with mock.patch("vc.views.distances_from_embeddings") as mock_distances:
        mock_distances.return_value = np.array([0.1, 0.2, 0.3])
        yield mock_distances
