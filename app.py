import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Please set it in your .env file or Render dashboard.")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Please set it in your .env file or Render dashboard.")

groq_client = Groq(api_key=GROQ_API_KEY)

# Hugging Face API setup for embeddings
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# CHAT_MODEL = "llama-3.1-8b-instant"  # Fast, free Groq model (Llama 3.1 8B)
CHAT_MODEL = "llama-3.1-8b-instant"
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

    conn = get_connection()
    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # Generate embeddings via Hugging Face API instead of local PyTorch
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": texts})
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error ({response.status_code}): {response.text}")
        
    return response.json()

@st.cache_data()
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()
    df["embedding"] = embed(df["comment"].tolist())
    df.to_parquet(CACHE_FILE)
    return df

st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {NEW_REVIEWS} reviews, answering with {CHAT_MODEL} (via Groq)")

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

    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

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