import pytest
from django.urls import reverse
from django.contrib.auth import get_user_model
from vc.models import Model, Expert, Conversation, Message, Document, clean_text
from django.core.files.uploadedfile import SimpleUploadedFile
import openai
from decouple import config

# from util import generate_pdf_content
from unittest import mock

openai.api_key = config("OPENAI_API_KEY")


def test_clean_text_removes_trailing_whitespace():
    assert clean_text("  This has trailing space  ") == "This has trailing space"


def test_clean_text_removes_leading_whitespace():
    assert clean_text("   This has leading space") == "This has leading space"


def test_clean_text_replaces_multiple_newlines():
    assert clean_text("Line 1\n\n\nLine 2") == "Line 1\nLine 2"


def test_clean_text_removes_empty_lines():
    assert clean_text("Line 1\n\n\n   \nLine 2") == "Line 1\nLine 2"


def test_clean_text_with_mixed_whitespace():
    assert clean_text("  Line 1  \n\n  Line 2  ") == "Line 1\nLine 2"


def test_clean_text_returns_empty_string():
    assert clean_text("    \n\n  ") == ""


def test_clean_text_returns_same_for_clean_input():
    assert clean_text("Already clean") == "Already clean"


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
