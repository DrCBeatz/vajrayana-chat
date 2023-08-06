from django.shortcuts import render
from .forms import QuestionForm
from decouple import config
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

expert = "Thrangu Rinpoche"
role = {
    "Thrangu Rinpoche": "prominent tulku (reincarnate lama) in the Kagyu school of Tibetan Buddhism.",
    "Mingyur Rinpoche": "Tibetan teacher and master of the Karma Kagyu and Nyingma lineages of Tibetan Buddhism.",
}

domain = "www.thrangu_rinpoche.com"
full_url = f"https://{domain}/"
openai.api_key = config("OPENAI_API_KEY")

MODEL = "gpt-3.5-turbo"
MAX_LEN = 1800
MAX_TOKENS = 300

DEBUG = True

previous_question = ""
previous_context = ""
previous_answer = ""

# load embeddings from parquet into dataframes

thrangu_rinpoche_df = pd.read_parquet("processed/thrangu_rinpoche_embeddings.parquet")

mingyur_rinpoche_df = pd.read_parquet("processed/mingyur_rinpoche_embeddings.parquet")


def create_context(question, df, max_len=MAX_LEN, size="ada"):
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
    model=MODEL,
    question=f"Who is {expert}?",
    max_len=MAX_LEN,
    size="ada",
    debug=False,
    max_tokens=MAX_TOKENS,
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
    global previous_question
    global previous_context
    global previous_answer
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
                    "content": f"You are the {expert}, {role[expert]} who answers questions about your life and Tibetan Buddhism.",
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
        return answer
    except Exception as e:
        print(e)
        return ""


def home(request):
    if request.htmx and request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data.get("question")
            if expert == "Thrangu Rinpoche":
                df = thrangu_rinpoche_df
            else:
                df = mingyur_rinpoche_df
            answer = answer_question(df, question=question, debug=DEBUG)
            return render(
                request, "answer.html", {"answer": answer, "question": question}
            )
    else:
        form = QuestionForm()
    return render(request, "home.html", {"form": form})


def get_title(request):
    global expert
    global previous_question
    global previous_context
    global previous_answer
    previous_question = previous_context = previous_answer = ""
    title = request.GET.get("title", "Thrangu Rinpoche")
    expert = title
    print(expert)
    return render(request, "_title.html", {"title": title})
