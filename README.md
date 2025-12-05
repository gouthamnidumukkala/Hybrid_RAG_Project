🚀 Hybrid Retrieval-Augmented Generation (RAG) System

Final Project – Advanced AI / CSC 790

This repository contains a Hybrid Retrieval-Augmented Generation (RAG) system built with:

FastAPI backend

BM25 Indexing (lexical retrieval)

Dense vector search using Chroma + bge-m3 embeddings

Reranking using bge-reranker

Local LLM generation using Ollama

Answer verification using Qwen / Llama models

SQuAD v1.1 dataset

This system retrieves relevant documents, generates answers, and checks for hallucinations.

📁 Project Structure
Mahi_Hybrid_RAG/
│
├── backend/
│   ├── app.py                 # FastAPI entry point
│   ├── bm25_indexer.py        # BM25 lexical search
│   ├── chroma_store.py        # Vector DB (Chroma)
│   ├── data_processor.py      # Dataset loading, preprocessing, chunking
│   ├── hybrid_retriever.py    # Hybrid BM25 + Embeddings retrieval
│   ├── answer_verifier.py     # LLM-based answer verification
│
├── config/
│   └── settings.py            # Configuration file
│
├── data/
│   └── SQuAD-v1.1.csv         # Dataset used for retrieval
│
├── evaluation/
│   ├── evaluate_system.py
│   ├── create_plots.py
│   └── run_evaluation.py
│
├── results/ (ignored by git)
├── requirements.txt
├── simple_rag.py
├── launch_rag_system.py
└── README.md

⚙️ Installation Instructions
1️⃣ Create a virtual environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

2️⃣ Install Ollama + required models

Download Ollama:
https://ollama.com/download

Pull models:

ollama pull qwen2.5:7b-instruct-q3_k_m
ollama pull bge-m3
ollama pull xitao/bge-reranker-v2-m3


Make sure Ollama is running.

3️⃣ Run the FastAPI backend

From the main project folder:

cd backend
python app.py


You'll see:

Uvicorn running on http://127.0.0.1:8000


Open Swagger UI:

👉 http://localhost:8000/docs

🧪 API Endpoints
🔍 POST /search

Hybrid document retrieval (BM25 + embeddings).

Example:

{
  "query": "earthquake"
}

🤖 POST /rag

Retrieval + LLM generation + answer verification.

Example:

{
  "query": "What happened during the 2008 Sichuan earthquake?"
}


Response includes:

retrieved documents

generated answer

verification score

hallucination risk

📘 Dataset Used

This project uses SQuAD v1.1, stored at:

data/SQuAD-v1.1.csv


During preprocessing:

Text is chunked

Lexical + vector indexes are built

Chroma embeddings are created

📊 Evaluation

Run evaluation script:

python run_evaluation.py


Produces:

retrieval metrics

generation metrics

verification accuracy

plots (if enabled)

Saved to:

results/
evaluation_results/

🧹 What is not included (ignored via .gitignore)

data/chroma_db/

data/*.pkl

results/

evaluation_results/

backend/*.log

__pycache__/

.env

venv/
