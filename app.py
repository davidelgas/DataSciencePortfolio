from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import torch
import requests
import os

app = Flask(__name__)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("docs.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Hugging Face API config
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_mistral(prompt):
    payload = {"inputs": prompt, "parameters": {"temperature": 0.7, "max_new_tokens": 200}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Error with the LLM response."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]
        query_embedding = embedder.encode([question], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k=3)
        retrieved_docs = "\n".join([documents[i] for i in indices[0]])

        full_prompt = f"Answer the following based on the context:\n\nContext:\n{retrieved_docs}\n\nQuestion: {question}\nAnswer:"
        response = query_mistral(full_prompt)

        return render_template("index.html", answer=response.strip(), question=question)

    return render_template("index.html", answer=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
