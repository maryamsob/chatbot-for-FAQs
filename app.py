# app.py

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# ------------------------
# Sample FAQ Data
# ------------------------
faqs = {
    "What is your name?": "I am your FAQ chatbot.",
    "How can I reset my password?": "You can reset your password by clicking on 'Forgot Password' on the login page.",
    "What are your working hours?": "We work from 9 AM to 6 PM, Monday to Friday.",
    "How can I contact support?": "You can contact support via email at support@example.com.",
}

# ------------------------
# Preprocessing
# ------------------------
stop_words = set([
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "is", "are", "was", "were", "the", "a", "an", "and", "or", "in",
    "on", "at", "to", "for", "with", "of", "from"
])

def preprocess(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Preprocess FAQ questions
processed_questions = [preprocess(q) for q in faqs.keys()]

# ------------------------
# Streamlit UI
# ------------------------
st.title("💬 FAQ Chatbot")
st.write("Ask me a question and I'll try to answer it!")

user_input = st.text_input("Enter your question:")

if user_input:
    processed_input = preprocess(user_input)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_questions + [processed_input])

    # Compute cosine similarity
    similarity = cosine_similarity(X[-1], X[:-1])
    max_idx = similarity.argmax()
    max_score = similarity[0, max_idx]

    # Threshold for response
    if max_score > 0.2:
        response = list(faqs.values())[max_idx]
    else:
        response = "Sorry, I don't know the answer to that."

    st.markdown(f"**Answer:** {response}")
