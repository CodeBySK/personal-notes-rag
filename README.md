# 🧠 Personal Notes RAG System (Local)

This project is a **Retrieval-Augmented Generation (RAG)** system that allows you to query your personal notes using a local LLM.

It reads text files, converts them into embeddings, stores them in a vector database, and uses a local model (via Ollama) to generate answers.

---

# 🚀 Features

* 📂 Load multiple `.txt` files
* ✂️ Chunk text for better retrieval
* 🧠 Generate embeddings using sentence-transformers
* 📦 Store embeddings in FAISS (vector database)
* 🔍 Retrieve relevant chunks based on query
* 🤖 Generate answers using local LLM (Ollama - Mistral)
* 💻 Fully local (no API cost)

---

# 🏗️ Architecture

```
Text Files → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone / Create Project

```bash
uv init personal-notes-rag
cd personal-notes-rag
```

---

## 2️⃣ Create Data Folder

```bash
mkdir data
```

Add your `.txt` files inside `data/`

---

## 3️⃣ Install Dependencies (using uv)

```bash
uv add sentence-transformers faiss-cpu requests
```

---

## 4️⃣ Install Ollama (Mac)

```bash
brew install ollama
```

Start server:

```bash
ollama serve
```

---

## 5️⃣ Download Model

```bash
ollama pull mistral
```

---

## 6️⃣ Run the App

```bash
uv run python main.py
```

---

# 🧪 Example Usage

```
Ask: What is normalization?
```

Output:

```
Normalization is the process of organizing data in a database...
```

---

# 📂 Project Structure

```
personal-notes-rag/
│
├── data/
│   ├── note1.txt
│   ├── note2.txt
│
├── main.py
├── pyproject.toml
└── README.md
```

---

# 🧠 How It Works

1. Load text files from `data/`
2. Split into smaller chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS
5. Convert user query into embedding
6. Retrieve similar chunks
7. Send context + query to LLM
8. Generate final answer

---

# ⚠️ Notes

* HuggingFace warning can be ignored
* FAISS runs locally (no cloud required)
* Ollama must be running (`ollama serve`)

---

# 🚀 Future Improvements

* Save & load FAISS index (avoid recomputation)
* Add Streamlit UI
* Add metadata filtering (file-based search)
* Improve chunking strategy
* Add chat history

---

# 💡 Tech Stack

* Python
* FAISS (Vector DB)
* sentence-transformers (Embeddings)
* Ollama (Local LLM)
* uv (Dependency management)

---

# 🔥 Key Learning

This project demonstrates:

* RAG pipeline design
* Vector search
* Embeddings
* Local LLM integration
* Real-world ML system thinking

---

# 🙌 Author

Built as part of learning **Modern Data Engineering + ML Systems**
