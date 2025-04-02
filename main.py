import streamlit as st
import pymupdf 
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Load embedding model (small, fast, and free)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Together.AI API Setup (Get free API key from https://www.together.ai/)
TOGETHER_API_KEY = "2d7d950b38e443bcc5a2b5547f630abb09f9f6b11ea78dd46498b9f42e56d680"
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf_doc:
        text += page.get_text("text") + "\n"
    return text

# Function to chunk text into small sections
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to create FAISS index
def create_faiss_index(chunks):
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Function to find relevant chunks using FAISS
def retrieve_relevant_chunks(query, chunks, index, k=3):
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    return " ".join([chunks[i] for i in indices[0]])

# Function to query Together.AI (Mistral-7B)
def query_togetherai(prompt):
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.7
    }
    response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["text"]

# Streamlit UI
st.title("📚 RAG Chatbot with Free Mistral-7B API")
st.write("Upload a PDF, ask questions, and get AI-powered answers!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    # Extract and process text
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, embeddings = create_faiss_index(chunks)

    # Store in session state
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index
    st.success("PDF uploaded and processed! Ask any question now.")

# Chat input
query = st.text_input("Ask a question based on the PDF:")
if query and "chunks" in st.session_state:
    # Retrieve relevant chunks
    relevant_text = retrieve_relevant_chunks(query, st.session_state["chunks"], st.session_state["index"])

    # Generate response using Mistral-7B
    final_prompt = f"Use the following context to answer: {relevant_text}\n\nQuestion: {query}\nAnswer:"
    response = query_togetherai(final_prompt)

    # Display result
    st.subheader("📢 AI Answer:")
    st.write(response)
