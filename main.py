import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


import requests

def generate_answer(query, context):
    prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    # DEBUG print (optional)
    #print("\nDEBUG RESPONSE:", data)

    return data.get("response", "No response from model")

def load_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def chunk_text(text):
    words = text.split()
    chunks = []
    size = 300
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks


documents = load_documents("data")

chunks = []
for doc in documents:
    parts = chunk_text(doc)
    for p in parts:
        chunks.append(p)

print("Chunks:", len(chunks))


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)


dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))


def ask(q):
    q_vec = model.encode([q])
    D, I = index.search(np.array(q_vec), 3)

    results = []
    for idx in I[0]:
        results.append(chunks[idx])

    return results


while True:
    q = input("Ask: ")

    if q == "exit":
        break

    results = ask(q)

    context = "\n".join(results)

    answer = generate_answer(q, context)

    print("\nAnswer:\n", answer)