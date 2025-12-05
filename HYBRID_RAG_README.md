# 🧠 Hybrid RAG System

A **Factual and Reliable Question Answering System** using Hybrid Retrieval-Augmented Generation with Quantized LLMs.

## ✨ Features

- **🔄 Hybrid Retrieval**: Combines BM25 (sparse) + BGE-M3 (dense) embeddings
- **🎯 Smart Reranking**: Uses BGE Reranker for improved result quality  
- **✅ Answer Verification**: Checks generated answers against source documents
- **🚨 Hallucination Detection**: Identifies and flags potentially inaccurate information
- **📊 Interactive UI**: Streamlit frontend with evaluation capabilities
- **⚡ Fast API**: RESTful backend for integration

## 🛠️ Models Used

- **Generator**: `qwen2.5:7b-instruct-q3_k_m` (Quantized 7B instruction-tuned model)
- **Retriever**: `bge-m3` (Multilingual embedding model)  
- **Reranker**: `xitao/bge-reranker-v2-m3` (Cross-encoder reranking)

## 📊 Dataset

- **Source**: SQuAD v1.1 (87,599 question-answer pairs)
- **Processed**: 18,894 document chunks with 512 token chunks
- **Indexing**: BM25 + ChromaDB vector store

## 🚀 Quick Start

### Prerequisites

1. **Ollama** must be running:
   ```bash
   ollama serve
   ```

2. **Required models** (should already be installed):
   ```bash
   ollama pull qwen2.5:7b-instruct-q3_k_m
   ollama pull bge-m3  
   ollama pull xitao/bge-reranker-v2-m3
   ```

### Launch System

```bash
# Single command to start everything
python launch_rag_system.py
```

This will start:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:8501  
- **API Documentation**: http://localhost:8000/docs

### Manual Launch (Alternative)

```bash
# Terminal 1: Start backend
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend  
cd frontend
streamlit run app.py --server.port 8501
```

## 🖥️ User Interface

### 💬 Chat Interface
- Ask questions and get verified answers
- Real-time answer verification
- View retrieved source documents
- Hallucination risk assessment

### 🔍 Document Search  
- Search knowledge base directly
- Compare BM25 vs Dense retrieval scores
- Fusion score visualization

### 📊 Answer Evaluation
- Evaluate answer quality and consistency
- Claim-by-claim verification analysis  
- Confidence scoring

## 🔧 API Endpoints

- `GET /health` - System health check
- `POST /search` - Document retrieval only
- `POST /generate` - Answer generation from context
- `POST /rag` - Complete RAG pipeline  
- `POST /evaluate` - Answer quality evaluation
- `GET /stats` - System statistics

## ⚙️ Configuration

Edit `config/settings.py` to adjust:

```python
RETRIEVAL_CONFIG = {
    "fusion_alpha": 0.3,      # BM25 vs Dense weight  
    "top_k_final": 5,         # Documents to return
    "rerank_top_k": 3         # Documents after reranking
}
```

## 📈 System Architecture

```
Query → Hybrid Retrieval → Generation → Verification → Response
        ↓                   ↓            ↓
    BM25 + Dense        Qwen2.5      Consistency 
    + Reranking         7B Model     Checker
```

## 🎯 Research Goals

This system implements the methodology from our research proposal:

1. **Hybrid Retrieval** - Combine sparse + dense methods
2. **Verification-Driven Generation** - Validate against sources  
3. **Hallucination Mitigation** - Detect and flag inaccuracies
4. **Human Evaluation Interface** - Quality assessment tools

## 📝 Evaluation Criteria

- **Relevance**: How well does the answer address the query?
- **Faithfulness**: Is the response grounded in evidence?
- **Hallucination**: Risk of fabricated information
- **Usefulness**: Overall helpfulness and completeness

## 🔍 Example Usage

```python
# Using the API directly
import requests

response = requests.post("http://localhost:8000/rag", json={
    "query": "What is the University of Notre Dame?",
    "top_k": 5,
    "include_verification": True
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Verification: {result['verification']['overall_verdict']}")
```

## 📊 Performance Stats

- **Documents**: 18,894 chunks from SQuAD
- **Vocabulary**: 96,926 unique terms  
- **Average Response Time**: ~2-3 seconds
- **Verification Accuracy**: Measured by human evaluation

## 🛠️ Development

### Project Structure
```
├── backend/           # FastAPI application
├── frontend/          # Streamlit interface  
├── config/            # Configuration files
├── data/              # Datasets and indices
├── launch_rag_system.py  # System launcher
└── README.md
```

### Key Components
- `hybrid_retriever.py` - Combines BM25 + dense retrieval
- `answer_verifier.py` - Checks answer consistency  
- `data_processor.py` - Processes SQuAD dataset
- `app.py` (backend) - FastAPI REST API
- `app.py` (frontend) - Streamlit UI

## 🤝 Contributing

This is a research implementation. Areas for improvement:

- [ ] Add more evaluation metrics
- [ ] Implement conversation memory
- [ ] Support for document upload
- [ ] Multi-language support  
- [ ] Advanced reranking strategies

## 📄 License

This project is for research and educational purposes.

---

**Built with ❤️ for reliable and factual question answering**