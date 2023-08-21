from django.shortcuts import render
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

openai.api_key = config("OPENAI_API_KEY")

expert = Expert.objects.first()


MODEL = Model.objects.first().name
MAX_LEN = 1800
MAX_TOKENS = 300

DEBUG = True


def get_embeddings(experts=Expert.objects.all()):
    print("Getting embeddings..")
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
                except:
                    print("No embeddings for " + document.title)
    return embeddings


def create_context(question, df, max_len=1800, size="ada"):
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for _, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
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

        response = openai.ChatCompletion.create(
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
        return ""


@login_required
def home(request):
    experts = Expert.objects.all()
    embeddings = get_embeddings(experts)

    if request.htmx and request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data.get("question")
            df = embeddings[expert.name]
            # Return both answer and context
            answer, context = answer_question(
                df,
                question=question,
                debug=DEBUG,
                conversation_id=request.session.get("conversation_id"),
            )

            if (
                "conversation_id" not in request.session
                or request.session["new_expert"]
            ):
                user = request.user
                conversation = Conversation.objects.create(
                    title=question, expert=expert, user=user
                )
                request.session["conversation_id"] = conversation.id
                if "new_expert" in request.session:
                    if request.session["new_expert"] == True:
                        request.session["new_expert"] = False
            else:
                # This is an existing session, get the current conversation
                conversation_id = request.session["conversation_id"]
                conversation = get_object_or_404(Conversation, id=conversation_id)

            message = Message.objects.create(
                conversation=conversation,
                question=question,
                answer=answer,
                context=context,
            )

            message.save()

            return render(
                request, "answer.html", {"answer": answer, "question": question}
            )
    else:
        form = QuestionForm()
    return render(
        request,
        "home.html",
        {"form": form, "experts": experts, "title": "Vajrayana AI Chat"},
    )


def change_expert(request):
    global expert

    title = request.GET.get("title", "Thrangu Rinpoche")
    expert = Expert.objects.get(name=title)
    # start new conversation if expert changes
    request.session["new_expert"] = True
    print(expert)
    return render(request, "_title.html", {"title": title})


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
    extra_context = {"title": f"Conversation with {expert}"}

    def get_context_data(self, **kwargs):
        context = super(ConversationDetailView, self).get_context_data(**kwargs)
        context["messages"] = Message.objects.filter(conversation=self.object).order_by(
            "id"
        )
        return context


class ConversationDeleteView(LoginRequiredMixin, ContextMixin, DeleteView):
    model = Conversation
    template_name = "conversation_confirm_delete.html"
    success_url = reverse_lazy("conversation-list")
    extra_context = {"title": "Delete Conversation"}
