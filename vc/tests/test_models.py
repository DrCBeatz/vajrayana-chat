import pytest
from django.urls import reverse
from django.contrib.auth import get_user_model
from vc.models import Model, Expert, Conversation, Message, Document
from django.core.files.uploadedfile import SimpleUploadedFile
import openai
from decouple import config
from reportlab.pdfgen import canvas
from io import BytesIO
from unittest import mock

openai.api_key = config("OPENAI_API_KEY")


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
def experts(db):
    model = Model.objects.create(name="gpt-3.5-turbo")

    return [
        Expert.objects.create(name="Expert1", model=model),
        Expert.objects.create(name="Expert2", model=model),
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


# Model tests


@pytest.mark.django_db
def test_create_model():
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    assert Model.objects.count() == 1
    assert model.name == "TestModel"


@pytest.mark.django_db
def test_create_expert():
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    expert = Expert.objects.create(
        name="TestExpert", prompt="This is a test prompt", model=model
    )
    assert Expert.objects.count() == 1
    assert expert.name == "TestExpert"


@pytest.mark.django_db
def test_create_expert():
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    expert = Expert.objects.create(
        name="TestExpert", prompt="This is a test prompt", model=model
    )
    assert Expert.objects.count() == 1
    assert expert.name == "TestExpert"
    assert expert.model == model


@pytest.mark.django_db
def test_create_conversation():
    User = get_user_model()
    user = User.objects.create_user(username="testuser", password="testpass123")
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    expert = Expert.objects.create(name="TestExpert", model=model)
    conversation = Conversation.objects.create(
        title="TestConversation", expert=expert, user=user
    )
    assert Conversation.objects.count() == 1
    assert conversation.title == "TestConversation"


@pytest.mark.django_db
def test_create_message():
    User = get_user_model()
    user = User.objects.create_user(username="testuser", password="testpass123")
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    expert = Expert.objects.create(name="TestExpert", model=model)
    conversation = Conversation.objects.create(
        title="TestConversation", expert=expert, user=user
    )
    message = Message.objects.create(
        conversation=conversation, question="TestQuestion", answer="TestAnswer"
    )
    assert Message.objects.count() == 1
    assert message.question == "TestQuestion"


def get_mock_response(content):
    """Helper function to mock HTTP responses."""
    response = mock.Mock()
    response.content = content.encode()
    return response


@pytest.mark.django_db
def test_document_save_with_html_content(experts):
    # Mock the external call to get the HTML content
    with mock.patch(
        "requests.get",
        return_value=get_mock_response("<html><body>Hello, World!</body></html>"),
    ):
        doc = Document(
            title="Test Doc", expert=experts[0], html_url="http://example.com"
        )
        doc.save()

        assert doc.content == "Hello, World!"


def test_has_content_changed_new_instance():
    doc = Document(title="Test Doc", content="Test content")
    assert doc._has_content_changed() is True
