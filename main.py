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

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure it is running on localhost:11434."
    except requests.exceptions.HTTPError as e:
        return f"Error: Ollama returned an unexpected status: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Request to Ollama failed: {e}"

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


def main():
    documents = load_documents("data")

    chunks = [p for doc in documents for p in chunk_text(doc)]

    print("Chunks:", len(chunks))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    def ask(q):
        q_vec = model.encode([q])
        _, I = index.search(np.array(q_vec), 3)
        return [chunks[idx] for idx in I[0] if idx != -1]

    while True:
        try:
            q = input("Ask: ")
        except EOFError:
            break

        if q == "exit":
            break

        results = ask(q)
        context = "\n".join(results)
        answer = generate_answer(q, context)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
