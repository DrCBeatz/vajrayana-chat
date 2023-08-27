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


# Fixtures


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


# views tests


@pytest.mark.django_db
def test_home_view_get(client):
    # Create a user and log in
    User = get_user_model()
    user = User.objects.create_user(username="testuser", password="testpass123")
    client.login(username="testuser", password="testpass123")

    # Create necessary objects for the view
    model = Model.objects.create(
        name="TestModel",
        input_token_cost=10.0,
        output_token_cost=5.0,
        context_length=20,
    )
    expert = Expert.objects.create(name="TestExpert", model=model)

    # GET request
    response = client.get(reverse("home"))

    # Test that the response is correct
    assert response.status_code == 200
    assert "form" in response.context


def test_expert_list_view(client, user, experts):
    client.login(username="testuser", password="testpassword")
    response = client.get("/experts/")
    assert response.status_code == 200
    assert len(response.context["experts"]) == 2
    assert "Expert1" in str(response.content)
    assert "Expert2" in str(response.content)


def test_expert_detail_view(client, user, experts):
    client.login(username="testuser", password="testpassword")
    expert = experts[0]
    response = client.get(f"/experts/{expert.pk}/")
    assert response.status_code == 200
    assert response.context["expert"] == expert


def test_expert_create_view(client, user):
    model = Model.objects.create(name="gpt-3.5-turbo")
    client.login(username="testuser", password="testpassword")

    response = client.post(
        "/experts/new/",
        data={
            "name": "NewExpert",
            "prompt": "This is a test prompt",
            "role": "This is a test role",
            "model": model.pk,
        },
    )

    assert response.status_code == 302
    assert Model.objects.count() == 1
    assert get_user_model().objects.count() == 1
    assert Expert.objects.count() == 1
    assert Expert.objects.first().name == "NewExpert"


def test_expert_update_view(client, user, experts):
    client.login(username="testuser", password="testpassword")
    expert = experts[0]
    response = client.post(
        f"/experts/{expert.pk}/edit/",
        data={
            "name": "UpdatedExpert",
            "prompt": expert.prompt,
            "role": expert.role,
            "model": expert.model.pk,
        },
    )
    assert response.status_code == 302
    expert.refresh_from_db()
    assert expert.name == "UpdatedExpert"


def test_expert_delete_view(client, user, experts):
    client.login(username="testuser", password="testpassword")
    expert = experts[0]
    response = client.post(f"/experts/{expert.pk}/delete/")
    assert response.status_code == 302
    assert Expert.objects.count() == 1


def test_conversation_list_view(client, user, conversations):
    client.login(username="testuser", password="testpassword")
    response = client.get("/conversations/")
    assert response.status_code == 200
    assert len(response.context["conversations"]) == 2
    assert conversations[0] in response.context["conversations"]
    assert conversations[1] in response.context["conversations"]


def test_conversation_detail_view(client, user, conversations, messages):
    client.login(username="testuser", password="testpassword")
    conversation = conversations[0]
    response = client.get(f"/conversations/{conversation.pk}/")
    assert response.status_code == 200
    assert response.context["object"] == conversation
    assert messages[0] in response.context["messages"]
    assert messages[1] in response.context["messages"]


def test_conversation_delete_view(client, user, conversations):
    client.login(username="testuser", password="testpassword")
    conversation = conversations[0]
    response = client.post(f"/conversations/{conversation.pk}/delete/")
    assert response.status_code == 302
    assert Conversation.objects.count() == 1


def test_document_list_view(client, user, documents):
    client.login(username="testuser", password="testpassword")
    response = client.get("/documents/")
    assert response.status_code == 200
    assert len(response.context["documents"]) == 2
    assert documents[0] in response.context["documents"]
    assert documents[1] in response.context["documents"]


def test_document_detail_view(client, user, documents):
    client.login(username="testuser", password="testpassword")
    document = documents[0]
    response = client.get(f"/documents/{document.pk}/")
    assert response.status_code == 200
    assert response.context["document"] == document


def test_document_delete_view(client, user, documents):
    client.login(username="testuser", password="testpassword")
    document = documents[0]
    response = client.post(f"/documents/{document.pk}/delete/")
    assert response.status_code == 302
    assert Document.objects.count() == 1


def test_document_create_view(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")
    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Document",
            "expert": expert.id,
            "embeddings": "embeddings/thrangu_rinpoche_embeddings.parquet",
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Document"


def test_document_create_view_content(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")
    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Document",
            "expert": expert.id,
            "content": "Test content",
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Document"
    assert Document.objects.first().content == "Test content"
    assert Document.objects.first().embeddings != ""


def test_document_create_view_txt_document(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")

    doc_content = "this is a test document"
    doc_file = SimpleUploadedFile("test_document.txt", doc_content.encode())

    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Document",
            "expert": expert.id,
            "document": doc_file,
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Document"
    assert Document.objects.first().content == doc_content
    assert Document.objects.first().embeddings != ""


def test_document_create_view_pdf_document(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")

    doc_content = "this is a test pdf document"
    pdf_buffer = generate_pdf_content(doc_content)

    pdf_file = SimpleUploadedFile("test_document.pdf", pdf_buffer.read())

    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Document",
            "expert": expert.id,
            "document": pdf_file,
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Document"
    assert Document.objects.first().content.replace("\n", "") == doc_content
    assert Document.objects.first().embeddings != ""


def test_document_create_view_html_url(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")

    website_content = """http://info.cern.ch
http://info.cern.ch - home of the first website
From here you can:
Browse the first website
Browse the first website using the line-mode browser simulator
Learn about the birth of the web
Learn about CERN, the physics laboratory where the web was born"""

    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Website",
            "expert": expert.id,
            "html_url": "http://info.cern.ch",
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Website"
    assert Document.objects.first().content == website_content
    assert Document.objects.first().embeddings != ""


def test_document_create_view_youtube_url(client, user, experts):
    expert = experts[0]
    client.login(username="testuser", password="testpassword")

    video_content = """thank you I go thank you very much thank you everyone thank you so much your lovely defending people thank you it's my privilege thank you"""

    response = client.post(
        "/documents/new/",
        data={
            "title": "Test Youtube Video",
            "expert": expert.id,
            "youtube_url": "https://www.youtube.com/watch?v=4nOSvpnCFTs",
        },
    )
    assert response.status_code == 302
    assert Document.objects.count() == 1
    assert Document.objects.first().title == "Test Youtube Video"
    assert Document.objects.first().content == video_content
    assert Document.objects.first().embeddings != ""


def test_document_update_view(client, user, experts, documents):
    document_to_update = documents[0]
    client.login(username="testuser", password="testpassword")
    updated_title = "Updated Title"
    response = client.post(
        f"/documents/{document_to_update.pk}/edit/",
        data={
            "title": updated_title,
            "expert": experts[1].id,
            "content": "Updated content",
        },
    )
    assert response.status_code == 302
    document_to_update.refresh_from_db()
    assert document_to_update.title == updated_title
    assert document_to_update.content == "Updated content"
    assert document_to_update.expert == experts[1]
    assert (
        document_to_update.embeddings
        != "embeddings/thrangu_rinpoche_embeddings.parquet"
    )
