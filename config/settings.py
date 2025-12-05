"""
Hybrid RAG Configuration
"""
import os

# Models Configuration
MODELS = {
    "generator": "qwen2.5:7b-instruct-q3_k_m",
    "retriever": "bge-m3", 
    "reranker": "xitao/bge-reranker-v2-m3"
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k_bm25": 10,
    "top_k_dense": 10,
    "top_k_final": 5,
    "fusion_alpha": 0.3,  # Weight for BM25 vs dense (α * BM25 + (1-α) * dense)
    "rerank_top_k": 3
}

# Database Configuration
DATABASE_CONFIG = {
    "chroma_db_path": "../data/chroma_db",
    "bm25_index_path": "../data/bm25_index.pkl",
    "squad_data_path": "../data/SQuAD-v1.1.csv"
}

# API Configuration
API_CONFIG = {
    "backend_host": "localhost",
    "backend_port": 8000,
    "frontend_host": "localhost", 
    "frontend_port": 8501,
    "ollama_base_url": "http://localhost:11434"
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "criteria": ["relevance", "faithfulness", "hallucination", "usefulness"],
    "rating_scale": [1, 2, 3, 4, 5],  # 1=Poor, 5=Excellent
    "hallucination_levels": ["None", "Minor", "Major"]
}