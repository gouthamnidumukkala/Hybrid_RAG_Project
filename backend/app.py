"""
FastAPI Backend for Hybrid RAG System
Provides REST API endpoints for the RAG pipeline
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
from hybrid_retriever import HybridRetriever
from answer_verifier import AnswerVerifier
import sys
import os

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import MODELS, RETRIEVAL_CONFIG, DATABASE_CONFIG, API_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid RAG API",
    description="Factual and Reliable Question Answering using Hybrid RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
hybrid_retriever = None
answer_verifier = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True

class GenerateRequest(BaseModel):
    query: str
    context_documents: Optional[List[Dict]] = None
    max_tokens: int = 500
    temperature: float = 0.7

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    max_tokens: int = 500
    temperature: float = 0.7
    include_verification: bool = True

class EvaluationRequest(BaseModel):
    query: str
    generated_answer: str
    retrieved_documents: List[Dict]
    criteria: List[str] = ["relevance", "faithfulness", "usefulness"]

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG components on startup"""
    global hybrid_retriever, answer_verifier
    
    try:
        print("Initializing Hybrid RAG components...")
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(
            bm25_index_path=DATABASE_CONFIG["bm25_index_path"],
            chroma_db_path=DATABASE_CONFIG["chroma_db_path"],
            fusion_alpha=RETRIEVAL_CONFIG["fusion_alpha"]
        )
        print("✓ Hybrid retriever initialized")
        
        # Initialize answer verifier
        answer_verifier = AnswerVerifier(generator_model=MODELS["generator"])
        print("✓ Answer verifier initialized")
        
        # Test Ollama connection
        response = requests.get(f"{API_CONFIG['ollama_base_url']}/api/tags")
        if response.status_code == 200:
            print("✓ Ollama connection verified")
        else:
            print("⚠ Warning: Ollama connection failed")
            
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise

def generate_answer_with_ollama(query: str, context: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Generate answer using Ollama model with context"""
    prompt = f"""You are a knowledgeable assistant that provides accurate, helpful answers based on the given context.

Context Information:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your response
4. Do not add information not present in the context
5. If you make any claims, they should be directly supported by the context

Answer:"""

    try:
        response = requests.post(
            f"{API_CONFIG['ollama_base_url']}/api/generate",
            json={
                "model": MODELS["generator"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": max_tokens
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'Failed to generate response')
        else:
            return f"Error generating answer: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hybrid RAG API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search - Retrieve relevant documents",
            "generate": "/generate - Generate answer from context",
            "rag": "/rag - Complete RAG pipeline",
            "evaluate": "/evaluate - Evaluate answer quality",
            "health": "/health - API health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama connection
        ollama_response = requests.get(f"{API_CONFIG['ollama_base_url']}/api/tags", timeout=5)
        ollama_status = ollama_response.status_code == 200
        
        # Check retriever status
        retriever_status = hybrid_retriever is not None
        verifier_status = answer_verifier is not None
        
        return {
            "status": "healthy" if all([ollama_status, retriever_status, verifier_status]) else "partial",
            "components": {
                "ollama": "online" if ollama_status else "offline",
                "hybrid_retriever": "loaded" if retriever_status else "not_loaded",
                "answer_verifier": "loaded" if verifier_status else "not_loaded"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search for relevant documents using hybrid retrieval"""
    try:
        if hybrid_retriever is None:
            raise HTTPException(status_code=500, detail="Hybrid retriever not initialized")
        
        results = hybrid_retriever.hybrid_search(
            query=request.query,
            top_k_bm25=RETRIEVAL_CONFIG["top_k_bm25"],
            top_k_dense=RETRIEVAL_CONFIG["top_k_dense"],
            final_top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_answer(request: GenerateRequest):
    """Generate answer from provided context documents"""
    try:
        # Prepare context from documents
        if request.context_documents:
            context = "\n\n".join([
                f"Document {i+1}: {doc.get('document', doc.get('text', ''))}"
                for i, doc in enumerate(request.context_documents)
            ])
        else:
            context = "No context provided."
        
        # Generate answer
        answer = generate_answer_with_ollama(
            query=request.query,
            context=context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "query": request.query,
            "answer": answer,
            "context_used": len(request.context_documents) if request.context_documents else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
async def complete_rag_pipeline(request: RAGRequest):
    """Complete RAG pipeline: retrieve, generate, and optionally verify"""
    try:
        if hybrid_retriever is None:
            raise HTTPException(status_code=500, detail="Hybrid retriever not initialized")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = hybrid_retriever.hybrid_search(
            query=request.query,
            top_k_bm25=RETRIEVAL_CONFIG["top_k_bm25"],
            top_k_dense=RETRIEVAL_CONFIG["top_k_dense"],
            final_top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        # Step 2: Prepare context
        context = "\n\n".join([
            f"Document {i+1}: {doc['document'][:800]}..."  # Truncate for context window
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Step 3: Generate answer
        answer = generate_answer_with_ollama(
            query=request.query,
            context=context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Step 4: Verify answer (optional)
        verification_result = None
        if request.include_verification and answer_verifier is not None:
            verification_result = answer_verifier.verify_answer(answer, retrieved_docs)
        
        return {
            "query": request.query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "verification": verification_result,
            "metadata": {
                "documents_retrieved": len(retrieved_docs),
                "reranking_used": request.use_reranking,
                "verification_included": request.include_verification,
                "fusion_alpha": RETRIEVAL_CONFIG["fusion_alpha"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_answer(request: EvaluationRequest):
    """Evaluate answer quality using verification system"""
    try:
        if answer_verifier is None:
            raise HTTPException(status_code=500, detail="Answer verifier not initialized")
        
        # Run verification
        verification_result = answer_verifier.verify_answer(
            answer=request.generated_answer,
            retrieved_documents=request.retrieved_documents
        )
        
        # Add confidence level
        confidence = answer_verifier.get_verification_confidence(verification_result)
        
        return {
            "query": request.query,
            "evaluation": verification_result,
            "confidence_level": confidence,
            "criteria_evaluated": request.criteria
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = {}
        
        if hybrid_retriever:
            stats["retrieval"] = hybrid_retriever.get_retrieval_stats()
        
        # Get model information
        try:
            response = requests.get(f"{API_CONFIG['ollama_base_url']}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                stats["available_models"] = [model["name"] for model in models_data.get("models", [])]
        except:
            stats["available_models"] = []
        
        stats["configuration"] = {
            "generator_model": MODELS["generator"],
            "retriever_model": MODELS["retriever"],
            "reranker_model": MODELS["reranker"],
            "fusion_alpha": RETRIEVAL_CONFIG["fusion_alpha"]
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=API_CONFIG["backend_host"],
        port=API_CONFIG["backend_port"],
        log_level="info"
    )