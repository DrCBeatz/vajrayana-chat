from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView,
)
from .forms import QuestionForm
from decouple import config
import pandas as pd
import openai
from openai.embeddings_utils import distances_from_embeddings
from .models import Model, Expert, Conversation, Message, Document
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from .timer_decorator import timer
from .mixins import ContextMixin
from django.core.cache import cache
from django.db.models import Max
from django.core.exceptions import ObjectDoesNotExist
import logging

logger = logging.getLogger(__name__)

openai.api_key = config("OPENAI_API_KEY")


MODEL = "gpt-3.5-turbo"
MAX_LEN = 1800
MAX_TOKENS = 300

DEBUG = True


def get_embeddings(experts=Expert.objects.all(), expert_name=None):
    if expert_name:
        experts = Expert.objects.filter(name=expert_name)

    last_modified_in_cache = cache.get("last_modified")
    latest_doc = Document.objects.latest("last_modified")

    num_documents_in_db = Document.objects.count()
    num_documents_in_cache = cache.get("num_documents", None)

    if num_documents_in_db != num_documents_in_cache:
        print("Number of documents has changed, regenerating embeddings..")
        embeddings = generate_embeddings_for_experts(experts)
        cache.set("embeddings", embeddings, None)
        cache.set("num_documents", num_documents_in_db, None)
    else:
        if expert_name:
            # Load only the embeddings for the specified expert
            embeddings = cache.get(f"embeddings_{expert_name}")

            if embeddings is None or (
                last_modified_in_cache
                and latest_doc.last_modified > last_modified_in_cache
            ):
                print(f"Getting embeddings for {expert_name}..")
                embeddings = generate_embeddings_for_experts(experts)

                cache.set(f"embeddings_{expert_name}", embeddings, None)
        else:
            # Load the embeddings for all experts
            embeddings = cache.get("embeddings")
            if embeddings is None or (
                last_modified_in_cache
                and latest_doc.last_modified > last_modified_in_cache
            ):
                print("Getting embeddings for all experts..")
                embeddings = generate_embeddings_for_experts(experts)
                cache.set("embeddings", embeddings, None)

    cache.set("last_modified", latest_doc.last_modified, None)
    return embeddings


def generate_embeddings_for_experts(experts):
    embeddings = {}
    documents = Document.objects.all()

    for e in experts:
        df_temp = pd.DataFrame()
        embeddings[e.name] = pd.DataFrame()

        for document in documents:
            if e.name == document.expert.name:
                try:
                    df_temp = pd.read_parquet(document.embeddings, engine="pyarrow")
                    embeddings[e.name] = pd.concat(
                        [
                            df_temp,
                            embeddings[e.name],
                        ],
                        ignore_index=True,
                    )
                except FileNotFoundError:
                    print(
                        f"File not found for document {document.title}, regenerating embeddings for {e.name}"
                    )
                except:
                    print(f"Unexpected error occurred for {document.title}")

    return embeddings


def create_context(question, df, max_len=1800, size="ada", distance_threshold=0.8):
    # Check if df is not a dataframe or if DataFrame is empty
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
        or df["embeddings"].isna().all()
        or df["text"].isna().all()
    ):
        return ""

    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for _, row in df.sort_values("distances", ascending=True).iterrows():
        if row["distances"] > distance_threshold:
            continue  # Skip this row, as it's not a close enough match

        cur_len += row["n_tokens"] + 4  # Adding 4 for the separators

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(
    request,
    df,
    openai_api=openai,
    conversation_id=None,
    model="gpt-3.5-turbo",
    question=f"Who is Thrangu Rinpoche?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=300,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    expert_id = request.session.get("expert", None)
    expert = Expert.objects.get(id=expert_id) if expert_id else Expert.objects.first()
    PROMPT = f"Answer the question based on the context below, in the first person as if you are {expert}."
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    previous_context = ""
    previous_question = ""
    previous_answer = ""

    if conversation_id:
        last_message = Message.objects.filter(conversation_id=conversation_id).last()
        if last_message:
            previous_context = last_message.context
            previous_question = last_message.question
            previous_answer = last_message.answer

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        prompt = f"""{PROMPT}
{"```Previous Context: " + previous_context + "```" if previous_context else ""}
```Current context: {context}```\n\n---\n\n
{"```Previous Answer: " + previous_answer + "```" if previous_answer else ""}
{"```Previous Question: " + previous_question + "```" if previous_question else ""}
```Current Question: {question}```            
\n Answer:"""

        if debug:
            print(f"\n***\n{prompt}\n***\n")

        response = openai_api.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are the {expert}, {expert.role} who answers questions about your life and Tibetan Buddhism.",
                },
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0,
        )

        previous_question = question
        previous_context = context
        answer = response["choices"][0]["message"]["content"].strip()
        previous_answer = answer
        return answer, context
    except Exception as e:
        print(e)
        return "", ""


@login_required
def load_expert_from_session(request):
    expert_id = request.session.get("expert", None)
    try:
        return (
            Expert.objects.get(id=int(expert_id))
            if expert_id
            else Expert.objects.first()
        )
    except (ObjectDoesNotExist, ValueError, TypeError):
        return Expert.objects.first()


def load_and_update_embeddings(experts):
    current_timestamps = {}
    cached_timestamps = cache.get("last_modified_timestamps", {})
    embeddings = {}

    for expert in experts:
        last_modified = Document.objects.filter(expert=expert).aggregate(
            Max("last_modified")
        )["last_modified__max"]

        if last_modified is not None:
            current_timestamps[expert.name] = last_modified

    for expert_name, timestamp in current_timestamps.items():
        if cached_timestamps.get(expert_name) != timestamp:
            new_embeddings = get_embeddings(expert_name=expert_name)
            if new_embeddings is not None:
                embeddings[expert_name] = new_embeddings[expert_name]
                cached_timestamps[expert_name] = timestamp
                cache.set(
                    f"embeddings_{expert_name}", new_embeddings[expert_name], None
                )

    if not embeddings:
        for expert in experts:
            cached_embedding = cache.get(f"embeddings_{expert.name}")
            if cached_embedding is not None:
                embeddings[expert.name] = cached_embedding

    cache.set("last_modified_timestamps", cached_timestamps, None)
    return embeddings


def handle_post_request(request, form, embeddings, current_expert):
    question = form.cleaned_data.get("question")

    try:
        df = embeddings[current_expert.name]
        if not isinstance(df, pd.DataFrame):
            print(
                f"Unexpected type for embeddings: {type(df)}, expert: {current_expert.name}"
            )
    except KeyError:
        df = None
        print(f"No embeddings found for {current_expert.name}.")

    if (
        df is None
        or (isinstance(df, pd.DataFrame) and df.empty)
        or (isinstance(df, dict) and not df)
    ):
        return render(
            request,
            "answer.html",
            {
                "answer": "No documents available for this expert.",
                "question": "N/A",
                "current_expert": current_expert,
            },
        )

    # Return both answer and context
    answer, context = answer_question(
        request,
        df,
        question=question,
        debug=False,  # Assuming DEBUG is a constant you've defined
        conversation_id=request.session.get("conversation_id"),
    )
    # Create a new conversation or use an existing one
    if "conversation_id" not in request.session or (
        "new_expert" in request.session and request.session["new_expert"]
    ):
        user = request.user
        conversation = Conversation.objects.create(
            title=question, expert=current_expert, user=user
        )
        request.session["conversation_id"] = conversation.id

        if "new_expert" in request.session and request.session["new_expert"]:
            request.session["new_expert"] = False
    else:
        conversation_id = request.session["conversation_id"]
        conversation = get_object_or_404(Conversation, id=conversation_id)

    # Create and save the message
    message = Message.objects.create(
        conversation=conversation,
        question=question,
        answer=answer,
        context=context,
    )

    message.save()

    return render(
        request,
        "answer.html",
        {"answer": answer, "question": question, "current_expert": current_expert},
    )


@login_required
def home(request):
    expert = load_expert_from_session(request)
    experts = Expert.objects.all()
    embeddings = load_and_update_embeddings(experts)

    if request.htmx and request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            return handle_post_request(request, form, embeddings, expert)
    else:
        form = QuestionForm()

    return render(
        request,
        "home.html",
        {
            "form": form,
            "experts": experts,
            "current_expert": expert,
            "title": "Vajrayana AI Chat",
        },
    )


def change_expert(request):
    title = request.GET.get("title", "Thrangu Rinpoche")
    try:
        new_expert = Expert.objects.get(name=title)
    except Expert.DoesNotExist:
        new_expert = None
        try:
            new_expert = Expert.objects.get(id=request.session.get("expert"))
        except (Expert.DoesNotExist, TypeError):
            new_expert = Expert.objects.get(
                name="Thrangu Rinpoche"
            )  # Fallback to a predefined expert
            title = "Thrangu Rinpoche"  # Update title to reflect the fallback expert
        logger.warning(f"Attempt to change to non-existent expert: {title}")

    request.session["expert"] = new_expert.id
    request.session["new_expert"] = True

    experts = Expert.objects.all()
    return render(
        request,
        "_title.html",
        {"title": title, "experts": experts, "current_expert": new_expert},
    )


class ExpertListView(LoginRequiredMixin, ContextMixin, ListView):
    model = Expert
    template_name = "expert_list.html"
    context_object_name = "experts"
    extra_context = {"title": "All Experts"}


class ExpertDetailView(LoginRequiredMixin, ContextMixin, DetailView):
    model = Expert
    template_name = "expert_detail.html"
    context_object_name = "expert"
    extra_context = {"title": "Expert Detail"}


class ExpertCreateView(LoginRequiredMixin, ContextMixin, CreateView):
    model = Expert
    template_name = "expert_form.html"
    fields = ["name", "prompt", "role", "model"]
    extra_context = {"title": "Create New Expert"}

    def get_success_url(self):
        return reverse_lazy("expert-list")


class ExpertUpdateView(LoginRequiredMixin, ContextMixin, UpdateView):
    model = Expert
    template_name = "expert_form.html"
    fields = ["name", "prompt", "role", "model"]
    success_url = reverse_lazy("expert-list")
    extra_context = {"title": "Edit Expert"}


class ExpertDeleteView(LoginRequiredMixin, ContextMixin, DeleteView):
    model = Expert
    template_name = "expert_confirm_delete.html"
    success_url = reverse_lazy("expert-list")
    extra_context = {"title": "Confirm Delete Expert"}


class DocumentListView(LoginRequiredMixin, ContextMixin, ListView):
    model = Document
    template_name = "document_list.html"
    context_object_name = "documents"
    extra_context = {"title": "All Documents"}


class DocumentDetailView(LoginRequiredMixin, ContextMixin, DetailView):
    model = Document
    template_name = "document_detail.html"
    context_object_name = "document"
    extra_context = {"title": "Document Detail"}


class DocumentCreateView(LoginRequiredMixin, ContextMixin, CreateView):
    model = Document
    template_name = "document_form.html"
    fields = [
        "title",
        "expert",
        "content",
        "document",
        "embeddings",
        "html_url",
        "youtube_url",
    ]
    extra_context = {"title": "Create New Document"}

    def get_success_url(self):
        return reverse_lazy("document-list")


class DocumentUpdateView(LoginRequiredMixin, ContextMixin, UpdateView):
    model = Document
    template_name = "document_form.html"
    fields = [
        "title",
        "expert",
        "content",
        "document",
        "embeddings",
        "html_url",
        "youtube_url",
    ]
    extra_context = {"title": "Update Document"}

    def get_success_url(self):
        return reverse_lazy("document-list")


class DocumentDeleteView(LoginRequiredMixin, ContextMixin, DeleteView):
    model = Document
    template_name = "document_confirm_delete.html"
    success_url = reverse_lazy("document-list")
    extra_context = {"title": "Delete Document"}


class ConversationListView(LoginRequiredMixin, ContextMixin, ListView):
    model = Conversation
    template_name = "conversation_list.html"
    extra_context = {"title": "List of Conversations"}
    context_object_name = "conversations"

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).order_by("-id")


class ConversationDetailView(LoginRequiredMixin, ContextMixin, DetailView):
    model = Conversation
    template_name = "conversation_detail.html"

    def get_context_data(self, **kwargs):
        context = super(ConversationDetailView, self).get_context_data(**kwargs)
        context["messages"] = Message.objects.filter(conversation=self.object).order_by(
            "id"
        )
        context["title"] = f"Conversation with {self.object.expert}"

        return context


class ConversationDeleteView(LoginRequiredMixin, ContextMixin, DeleteView):
    model = Conversation
    template_name = "conversation_confirm_delete.html"
    success_url = reverse_lazy("conversation-list")
    extra_context = {"title": "Delete Conversation"}
