import pytest
from django.test import TestCase, RequestFactory
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from vc.models import Model, Expert, Conversation, Message, Document
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.sessions.middleware import SessionMiddleware
import openai
from vc.views import (
    change_expert,
    generate_embeddings_for_experts,
    load_expert_from_session,
    load_and_update_embeddings,
)
from decouple import config
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from django.core.cache import cache
from conftest import generate_pdf_content

openai.api_key = config("OPENAI_API_KEY")


# views tests


@pytest.mark.django_db
def test_load_and_update_embeddings(experts):
    expert1, expert2 = experts

    doc1 = Document.objects.create(
        title="Document1", expert=expert1, content="Content1"
    )
    doc2 = Document.objects.create(
        title="Document2", expert=expert2, content="Content2"
    )

    with patch("vc.views.get_embeddings") as mock_get_embeddings, patch.object(
        Document, "embed"
    ) as mock_embed_method:
        mock_get_embeddings.return_value = {
            "Expert1": "Embeddings1",
            "Expert2": "Embeddings2",
        }
        mock_embed_method.return_value = None

        # Updating contentent and saving to trigger embed()
        doc1.content = "Updated Content"
        doc1.save()

        cache.clear()

        result = load_and_update_embeddings([expert1, expert2])

        cached_timestamps = cache.get("last_modified_timestamps", {})
    assert cached_timestamps == {
        "Expert1": Document.objects.filter(expert=expert1).first().last_modified,
        "Expert2": Document.objects.filter(expert=expert2).first().last_modified,
    }

    assert result == {
        "Expert1": "Embeddings1",
        "Expert2": "Embeddings2",
    }

    mock_embed_method.assert_any_call()


@pytest.mark.django_db
def test_load_and_update_embeddings_uses_cache(experts):
    def mock_cache_get_function(key, default=None):
        return {
            "last_modified_timestamps": {
                "Expert1": Document.objects.filter(expert=expert1)
                .first()
                .last_modified,
                "Expert2": Document.objects.filter(expert=expert2)
                .first()
                .last_modified,
            },
            "embeddings_Expert1": "OldEmbeddings1",
            "embeddings_Expert2": "OldEmbeddings2",
        }.get(key, default)

    expert1, expert2 = experts
    # Setup
    Document.objects.create(title="Document1", expert=expert1, content="Content1")
    Document.objects.create(title="Document2", expert=expert2, content="Content2")

    with patch("vc.views.get_embeddings") as mock_get_embeddings, patch.object(
        Document, "embed"
    ) as mock_embed_method, patch(
        "django.core.cache.cache.get"
    ) as mock_cache_get, patch(
        "django.core.cache.cache.set"
    ) as mock_cache_set:
        mock_get_embeddings.return_value = {
            "Expert1": "Embeddings1",
            "Expert2": "Embeddings2",
        }

        # Mock the cache to return pre-existing timestamps and embeddings
        mock_cache_get.side_effect = mock_cache_get_function

        result = load_and_update_embeddings([expert1, expert2])

        # Assertions
        # mock_cache_get.assert_any_call("last_modified_timestamps")

        mock_cache_set.assert_called()  # Add more details to this assertion as per your use-case
        assert result == {
            "Expert1": "OldEmbeddings1",  # Should use cached embeddings
            "Expert2": "OldEmbeddings2",  # Should use cached embeddings
        }


@pytest.mark.django_db
def test_load_expert_from_session(expert_obj, user):
    request = RequestFactory().get("/")
    request.user = user
    request.session = {"expert": expert_obj.id}

    result = load_expert_from_session(request)

    assert result == expert_obj


@pytest.mark.django_db
def test_load_expert_from_session_no_user(expert_obj):
    request = RequestFactory().get("/")
    request.user = AnonymousUser()
    request.session = {"expert": expert_obj.id}
    result = load_expert_from_session(request)
    assert result.status_code == 302
    assert result.url == "/accounts/login/?next=/"


@pytest.mark.django_db
def test_load_expert_from_session_no_expert(expert_obj, user):
    request = RequestFactory().get("/")
    request.user = user
    request.session = {}

    result = load_expert_from_session(request)
    assert result == Expert.objects.first()


@pytest.mark.django_db
def test_load_expert_from_session_invalid_expert_id(expert_obj, user):
    request = RequestFactory().get("/")
    request.user = user
    request.session = {"expert": "invalid_id"}

    result = load_expert_from_session(request)
    assert result == Expert.objects.first()


@pytest.mark.django_db
def test_load_expert_from_session_invalid_but_well_formed_expert_id(expert_obj, user):
    request = RequestFactory().get("/")
    request.user = user
    request.session = {"expert": 999}
    result = load_expert_from_session(request)
    assert result == Expert.objects.first()


@pytest.mark.django_db
def test_load_expert_from_session_multiple_experts(experts, user):
    request = RequestFactory().get("/")
    request.user = user
    request.session = {"expert": experts[1].id}
    result = load_expert_from_session(request)
    assert result == experts[1]
    assert result != experts[0]


@pytest.mark.django_db
def test_load_expert_from_session_empty_db_and_session(user):
    """
    Test to ensure that load_expert_from_session returns None when both
    the database and session are empty. i.e.:
    Expert.objects.first() == None
    """
    request = RequestFactory().get("/")
    request.user = user
    request.session = {}
    result = load_expert_from_session(request)
    assert result == None


def test_generate_embeddings_for_experts(db, experts, documents2):
    # Arrange: Setup is done by fixtures

    # Act
    result = generate_embeddings_for_experts(experts)

    # Assert
    assert "Expert1" in result
    assert "Expert2" in result
    assert isinstance(result["Expert1"], pd.DataFrame)
    assert isinstance(result["Expert2"], pd.DataFrame)
    assert result["Expert1"].shape == (1, 2)
    assert result["Expert2"].shape == (1, 2)
    assert np.allclose(result["Expert1"].embedding.values[0], [0.1, 0.2])
    assert np.allclose(result["Expert2"].embedding.values[0], [0.3, 0.4])


@pytest.mark.django_db
def test_get_embeddings(experts):
    from vc.views import get_embeddings

    expert1 = experts[0]
    expert2 = experts[1]
    Document.objects.create(
        title="Doc1",
        expert=expert1,
        content="This is a test document",
        embeddings="embeddings/thrangu_rinpoche_embeddings.parquet",
    )

    Document.objects.create(
        title="Doc2",
        expert=expert2,
        content="This is another test document",
        embeddings="embeddings/mingyur_rinpoche_embeddings.parquet",
    )

    result = get_embeddings()

    assert "Expert1" in result
    assert "Expert2" in result


@pytest.mark.django_db
def test_home_view_authenticated(client, user, expert_obj, document_obj):
    client.force_login(user)
    url = reverse("home")
    response = client.get(url)
    assert response.status_code == 200
    assert b"Vajrayana AI Chat" in response.content
    assert b"Expert1" in response.content


@pytest.mark.django_db
def test_home_view_unauthenticated(client, user, expert_obj, document_obj):
    url = reverse("home")
    response = client.get(url)
    assert response.status_code == 302  # HTTP status code for redirection
    assert response.url == "/accounts/login/?next=/"


class ChangeExpertViewTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.middleware = SessionMiddleware(lambda req: None)
        self.model = Model.objects.create(
            name="gpt-3.5-turbo",
            context_length=4096,
            input_token_cost=0.0015,
            output_token_cost=0.002,
        )
        self.expert1 = Expert.objects.create(
            name="Expert1", prompt="Prompt1", role="Role1", model=self.model
        )
        self.expert2 = Expert.objects.create(
            name="Expert2", prompt="Prompt2", role="Role2", model=self.model
        )
        self.fallback_expert = Expert.objects.create(
            name="Thrangu Rinpoche",
            prompt="PromptFallback",
            role="RoleFallback",
            model=self.model,
        )

    def test_change_expert_view(self):
        request = self.factory.get(
            reverse("change_expert") + f"?title={self.expert1.name}"
        )

        # Apply session middleware manually to add session support to the request
        self.middleware.process_request(request)
        request.session.save()

        response = change_expert(request)
        self.assertEqual(response.status_code, 200)

        # Validate that the session stores the expected expert ID
        self.assertEqual(request.session["expert"], self.expert1.id)

    def test_change_expert_with_invalid_title(self):
        request = self.factory.get(
            reverse("change_expert") + "?title=InvalidExpertName"
        )

        # Apply session middleware manually to add session support to the request
        self.middleware.process_request(request)
        request.session.save()

        response = change_expert(request)
        self.assertEqual(response.status_code, 200)

        # Assume session is the way you're storing the current expert
        self.assertEqual(request.session["expert"], self.fallback_expert.id)

        # Additional checks can go here, for example, you can check whether the content of the response is as expected
        self.assertIn(
            b"Thrangu Rinpoche", response.content
        )  # Replace with actual check

    def test_change_expert_view_missing_title(self):
        request = self.factory.get(reverse("change_expert"))

        # Apply session middleware manually to add session support to the request
        self.middleware.process_request(request)
        request.session.save()

        response = change_expert(request)
        self.assertEqual(response.status_code, 200)

        # Validate that the session stores the fallback expert when no title is provided
        self.assertEqual(request.session["expert"], self.fallback_expert.id)

        # Check if the response contains the name of the fallback expert
        self.assertIn(b"Thrangu Rinpoche", response.content)


def test_expert_list_view(client_with_user, experts):
    url = reverse("expert-list")
    client, user = client_with_user
    response = client.get(url)
    assert response.status_code == 200
    assert len(response.context["experts"]) == 2
    assert "Expert1" in str(response.content)
    assert "Expert2" in str(response.content)


def test_expert_detail_view(client_with_user, experts):
    expert = Expert.objects.get(name="Expert1")
    url = reverse("expert-detail", args=[str(expert.id)])
    client, user = client_with_user
    response = client.get(url)
    assert response.status_code == 200
    assert response.context["expert"] == expert
    assert "Expert1" in str(response.content)
    assert "Prompt1" in str(response.content)
    assert "Role1" in str(response.content)


def test_expert_create_view(client_with_user, model_obj):
    url = reverse("expert-create")
    client, user = client_with_user

    response = client.post(
        url,
        {
            "name": "Expert3",
            "prompt": "Prompt3",
            "role": "Role3",
            "model": model_obj.id,
        },
    )

    assert response.status_code == 302
    assert Expert.objects.count() == 1
    assert get_user_model().objects.count() == 1
    assert Expert.objects.count() == 1
    assert Expert.objects.first().name == "Expert3"
    assert Expert.objects.first().prompt == "Prompt3"
    assert Expert.objects.first().role == "Role3"


def test_expert_update_view(client_with_user, model_obj):
    expert = Expert.objects.create(
        name="Expert1",
        prompt="Prompt1",
        role="Role1",
        model=model_obj,
    )

    url = reverse("expert-update", args=[expert.id])
    client, user = client_with_user
    response = client.post(
        url,
        {
            "name": "UpdatedExpert",
            "prompt": "UpdatedPrompt",
            "role": "UpdatedRole",
            "model": model_obj.id,
        },
    )

    expert.refresh_from_db()  # Refresh the expert object to get updated values

    assert response.status_code == 302
    assert expert.name == "UpdatedExpert"
    assert expert.prompt == "UpdatedPrompt"
    assert expert.role == "UpdatedRole"


def test_expert_delete_view(client_with_user, model_obj):
    expert = Expert.objects.create(
        name="Expert1",
        prompt="Prompt1",
        role="Role1",
        model=model_obj,
    )

    url = reverse("expert-delete", args=[expert.id])
    client, user = client_with_user

    response = client.post(url)

    assert response.status_code == 302
    assert Expert.objects.count() == 0


def test_conversation_list_view(client_with_user, expert_obj):
    client, user = client_with_user
    # Create some conversations
    Conversation.objects.create(title="Conversation1", expert=expert_obj, user=user)
    Conversation.objects.create(title="Conversation2", expert=expert_obj, user=user)

    url = reverse("conversation-list")
    response = client.get(url)

    assert response.status_code == 200
    assert len(response.context["conversations"]) == 2
    assert "Conversation1" in str(response.content)
    assert "Conversation2" in str(response.content)


def test_conversation_detail_view(client_with_user, expert_obj):
    client, user = client_with_user
    # Create a conversation
    conversation = Conversation.objects.create(
        title="Conversation1", expert=expert_obj, user=user
    )
    # create some messages for the conversation
    Message.objects.create(
        conversation=conversation,
        question="Question1",
        answer="Answer1",
        context="Context1",
    )
    Message.objects.create(
        conversation=conversation,
        question="Question2",
        answer="Answer2",
        context="Context2",
    )

    url = reverse("conversation-detail", args=[conversation.id])
    response = client.get(url)

    assert response.status_code == 200
    assert response.context["conversation"] == conversation
    assert response.context["messages"].count() == 2
    assert "Question1" in str(response.content)
    assert "Answer1" in str(response.content)
    assert "Question2" in str(response.content)
    assert "Answer2" in str(response.content)


def test_conversation_delete_view(client_with_user, expert_obj):
    client, user = client_with_user
    # Create a conversation
    conversation = Conversation.objects.create(
        title="Conversation1", expert=expert_obj, user=user
    )

    url = reverse("conversation-delete", args=[conversation.id])
    response = client.post(url)

    assert response.status_code == 302
    assert Conversation.objects.count() == 0


def test_document_list_view(client_with_user, document_obj):
    client, user = client_with_user
    url = reverse("document-list")
    response = client.get(url)

    assert response.status_code == 200
    assert len(response.context["documents"]) == 1
    assert document_obj in response.context["documents"]
    assert "Document1" in str(response.content)


def test_document_detail_view(client_with_user, document_obj):
    client, user = client_with_user
    url = reverse("document-detail", kwargs={"pk": document_obj.pk})
    response = client.get(url)

    assert response.status_code == 200
    assert response.context["document"] == document_obj
    assert "Document1" in str(response.content)


def test_document_delete_view(client_with_user, document_obj):
    client, user = client_with_user
    url = reverse("document-delete", kwargs={"pk": document_obj.pk})

    # Make a GET request to confirm page exists
    response = client.get(url)
    assert response.status_code == 200

    # Make a POST request to delete the document
    response = client.post(url)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    # Confim the document is deleted
    assert Document.objects.filter(pk=document_obj.pk).count() == 0


def test_document_create_view_content(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")

    post_data = {
        "title": "Document1",
        "expert": expert_obj.id,
        "content": "This is a test document",
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.title == "Document1"
    assert document.expert == expert_obj
    assert document.content == "This is a test document"


def test_document_create_view_embeddings(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")

    post_data = {
        "title": "Document1",
        "expert": expert_obj.id,
        "embeddings": "embeddings/thrangu_rinpoche_embeddings.parquet",
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.title == "Document1"
    assert document.expert == expert_obj
    assert document.embeddings != None


def test_document_create_view_txt_document(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")

    doc_content = "this is a test document"
    doc_file = SimpleUploadedFile("test_document.txt", doc_content.encode())

    post_data = {
        "title": "Document1",
        "expert": expert_obj.id,
        "document": doc_file,
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.title == "Document1"
    assert document.expert == expert_obj
    assert document.content == doc_content
    assert Document.objects.first().embeddings != ""


def test_document_create_view_pdf_document(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")

    doc_content = "this is a test pdf document"
    pdf_buffer = generate_pdf_content(doc_content)

    pdf_file = SimpleUploadedFile("test_document.pdf", pdf_buffer.read())

    post_data = {
        "title": "Document1",
        "expert": expert_obj.id,
        "document": pdf_file,
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.title == "Document1"
    assert document.expert == expert_obj
    assert Document.objects.first().content.replace("\n", "") == doc_content
    assert Document.objects.first().embeddings != ""


def test_document_create_html_url(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")
    website_content = """http://info.cern.ch
http://info.cern.ch - home of the first website
From here you can:
Browse the first website
Browse the first website using the line-mode browser simulator
Learn about the birth of the web
Learn about CERN, the physics laboratory where the web was born"""

    post_data = {
        "title": "Test Website",
        "expert": expert_obj.id,
        "html_url": "http://info.cern.ch",
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.title == "Test Website"
    assert document.expert == expert_obj
    assert document.content == website_content
    assert Document.objects.first().embeddings != ""


def test_document_create_youtube_url(client_with_user, expert_obj):
    client, user = client_with_user
    url = reverse("document-create")
    video_content = """thank you I go thank you very much thank you everyone thank you so much your lovely defending people thank you it's my privilege thank you"""

    post_data = {
        "title": "Test Youtube Video",
        "expert": expert_obj.id,
        "youtube_url": "https://www.youtube.com/watch?v=4nOSvpnCFTs",
    }

    response = client.post(url, data=post_data)
    assert response.status_code == 302
    assert response.url == reverse("document-list")

    assert Document.objects.count() == 1
    document = Document.objects.first()
    assert document.expert == expert_obj
    assert Document.objects.first().title == "Test Youtube Video"
    assert Document.objects.first().content == video_content
    assert Document.objects.first().embeddings != ""


@pytest.mark.django_db
def test_document_updated_view(client_with_user, document_obj):
    client, user = client_with_user
    url = reverse("document-update", kwargs={"pk": document_obj.pk})

    # Test GET request (Retrieving the form)
    response = client.get(url)
    assert response.status_code == 200

    # Test POST request (Submitting the form)
    new_title = "Document1 Updated"
    response = client.post(
        url,
        {
            "title": new_title,
            "expert": document_obj.expert.id,
            "content": document_obj.content,
        },
    )

    assert response.status_code == 302
    assert response.url == reverse("document-list")

    document_obj.refresh_from_db()
    assert document_obj.title == new_title
