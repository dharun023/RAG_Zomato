import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
import ollama
from dotenv import load_dotenv

load_dotenv()
# os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"


# client = ollama.Client(host="http://127.0.0.1:11434")

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "https://ollama.com",
)

client = ollama.Client(
    host=OLLAMA_HOST,
    headers={
        "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
    },
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")




EMBEDDING_MODEL = "nomic-embed-text"   # or "llama3.2" if you prefer
CHAT_MODEL = "llama3.2"                # e.g. "llama3.1", "llama3.2", etc.
NEW_REVIEWS = 500
TOK_K = 5
CACHE_FILE = "review_embeddings.parquet"


def read_reviews_from_snowflake():
    def get_connection():
        return snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df


def embed(texts: list[str]):
    # Ollama's batched embed API
    result = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return result["embeddings"]


@st.cache_data()
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()
    df["embedding"] = embed(df["comment"].tolist())
    df.to_parquet(CACHE_FILE)
    return df


st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {NEW_REVIEWS} reviews, answering with {CHAT_MODEL} (local Llama via Ollama)")


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def find_similar_reviews(question, df):
    question_vector = embed([question])[0]

    scores = []
    for review_vector in df["embedding"]:
        scores.append(cosine_similarity(question_vector, review_vector))

    df = df.copy()
    df["score"] = scores
    return df.nlargest(TOK_K, "score")


def ask_llm(question, top_reviews):
    context = ""
    for _, row in top_reviews.iterrows():
        context += f" ({row['city']}, {row['rating']} stars) {row['comment']}\n"

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't cover it, say so."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nReviews:\n{context}"}
    ]

    response = ollama.chat(model=CHAT_MODEL, messages=messages)
    return response["message"]["content"]


review_df = load_reviews()

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?"
)

if question:
    top_reviews = find_similar_reviews(question, review_df)
    answer = ask_llm(question, top_reviews)

    st.markdown("**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[["city", "rating", "comment"]], hide_index=True)