import os

import numpy as np
import ollama
import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def required_env(name):
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value.strip()


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "https://ollama.com",
).strip()

OLLAMA_API_KEY = required_env("OLLAMA_API_KEY")

CHAT_MODEL = required_env("OLLAMA_MODEL")
EMBEDDING_MODEL = required_env("EMBEDDING_MODEL")

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
    },
)


NEW_REVIEWS = 500
TOP_K = 5


st.set_page_config(
    page_title="Zomato Review Analytics",
    page_icon="🍽️",
    layout="wide",
)


def get_connection():
    return snowflake.connector.connect(
        user=required_env("SNOWFLAKE_USER"),
        password=required_env("SNOWFLAKE_PASSWORD"),
        account=required_env("SNOWFLAKE_ACCOUNT"),
        warehouse=required_env("SNOWFLAKE_WAREHOUSE"),
        database=required_env("SNOWFLAKE_DATABASE"),
        schema=required_env("SNOWFLAKE_SCHEMA"),
    )


@st.cache_data(ttl=600)
def read_reviews_from_snowflake():
    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            df = cursor.fetch_pandas_all()
        finally:
            cursor.close()
    finally:
        connection.close()

    df.columns = [column.lower() for column in df.columns]
    df["comment"] = df["comment"].fillna("").astype(str)

    return df


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not texts:
        return []

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response["embeddings"]


@st.cache_data(ttl=3600)
def create_review_embeddings(review_texts):
    return embed(list(review_texts))


@st.cache_data(ttl=600)
def load_reviews():
    df = read_reviews_from_snowflake()

    embeddings = create_review_embeddings(
        tuple(df["comment"].tolist())
    )

    if len(embeddings) != len(df):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of reviews."
        )

    df = df.copy()
    df["embedding"] = embeddings

    return df


def cosine_similarity(vec_a, vec_b):
    vec_a = np.asarray(vec_a, dtype=np.float32)
    vec_b = np.asarray(vec_b, dtype=np.float32)

    denominator = (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / denominator)


def find_similar_reviews(question, df):
    question_vector = embed([question])[0]

    scores = [
        cosine_similarity(question_vector, review_vector)
        for review_vector in df["embedding"]
    ]

    result = df.copy()
    result["score"] = scores

    return result.nlargest(TOP_K, "score")


def ask_llm(question, top_reviews):
    context_lines = []

    for _, row in top_reviews.iterrows():
        context_lines.append(
            f"({row['city']}, {row['rating']} stars) "
            f"{row['comment']}"
        )

    context = "\n".join(context_lines)

    system_prompt = (
        "Answer only using the customer reviews provided. "
        "Be concise. If the reviews do not cover the question, "
        "say that the reviews do not provide enough information."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Reviews:\n{context}"
            ),
        },
    ]

    response = client.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={
            "temperature": 0,
        },
    )

    return response.message.content


st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews using "
    f"{CHAT_MODEL} and {EMBEDDING_MODEL}"
)

with st.sidebar:
    st.write("Configuration")
    st.write({
        "ollama_host": OLLAMA_HOST,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "api_key_configured": bool(OLLAMA_API_KEY),
    })

try:
    review_df = load_reviews()

except Exception as error:
    st.error("Could not load or embed reviews.")
    st.exception(error)
    st.stop()


question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common complaints "
        "about delivery?"
    ),
)


if question.strip():
    try:
        top_reviews = find_similar_reviews(
            question,
            review_df,
        )

        answer = ask_llm(
            question,
            top_reviews,
        )

        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander(
            "Reviews used to build this answer"
        ):
            st.dataframe(
                top_reviews[
                    ["city", "rating", "comment", "score"]
                ],
                hide_index=True,
            )

    except Exception as error:
        st.error("Could not answer the question.")
        st.exception(error)