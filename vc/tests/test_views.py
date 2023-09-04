import pytest
from django.test import TestCase, RequestFactory
from django.urls import reverse
from django.contrib.auth import get_user_model
from vc.models import Model, Expert, Conversation, Message, Document
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.sessions.middleware import SessionMiddleware
import openai
from vc.views import change_expert, generate_embeddings_for_experts
from decouple import config
from reportlab.pdfgen import canvas
from io import BytesIO
from unittest.mock import Mock
import pandas as pd
import numpy as np
from django.core.cache import cache
from django.core.files import File


openai.api_key = config("OPENAI_API_KEY")


def generate_pdf_content(content="this is a test pdf document"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(100, 100))
    c.drawString(10, 10, content)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# Fixtures


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


# views tests


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


def test_conversation_Detail_view(client_with_user, expert_obj):
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
