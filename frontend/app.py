"""
Streamlit Frontend for Hybrid RAG System
Interactive UI for question answering with evaluation capabilities
"""
import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .verification-good {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.5rem;
    }
    .verification-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.5rem;
    }
    .verification-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.5rem;
    }
    .document-card {
        background: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API backend is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def search_documents(query: str, top_k: int = 5, use_reranking: bool = True):
    """Search for documents using the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "use_reranking": use_reranking
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def complete_rag_pipeline(query: str, top_k: int = 5, use_reranking: bool = True, 
                         max_tokens: int = 500, temperature: float = 0.7, 
                         include_verification: bool = True):
    """Run complete RAG pipeline"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag",
            json={
                "query": query,
                "top_k": top_k,
                "use_reranking": use_reranking,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "include_verification": include_verification
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"RAG pipeline failed: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"RAG pipeline error: {str(e)}")
        return None

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_verification_results(verification: Dict):
    """Display verification results with color coding"""
    if not verification:
        return
    
    overall_verdict = verification.get('overall_verdict', 'UNKNOWN')
    consistency_score = verification.get('consistency_score', 0)
    hallucination_risk = verification.get('hallucination_risk', 'UNKNOWN')
    
    # Color coding based on results
    if overall_verdict in ['SUPPORTED'] and consistency_score >= 0.7:
        css_class = "verification-good"
        status_emoji = "✅"
    elif overall_verdict in ['PARTIALLY_SUPPORTED'] or consistency_score >= 0.4:
        css_class = "verification-warning"
        status_emoji = "⚠️"
    else:
        css_class = "verification-danger"
        status_emoji = "❌"
    
    st.markdown(f"""
    <div class="{css_class}">
        <h4>{status_emoji} Answer Verification</h4>
        <p><strong>Overall Verdict:</strong> {overall_verdict}</p>
        <p><strong>Consistency Score:</strong> {consistency_score:.3f}</p>
        <p><strong>Hallucination Risk:</strong> {hallucination_risk}</p>
        <p><strong>Summary:</strong> {verification.get('summary', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Hybrid RAG System</h1>', unsafe_allow_html=True)
    st.markdown("*Factual and Reliable Question Answering using Quantized LLMs*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Health Check
        health_status = check_api_health()
        if health_status["status"] == "healthy":
            st.success("✅ API Backend Online")
        else:
            st.error(f"❌ API Backend: {health_status.get('message', 'Offline')}")
            st.stop()
        
        # System Stats
        if st.button("📊 System Stats"):
            stats = get_system_stats()
            if stats:
                st.json(stats)
        
        st.markdown("---")
        
        # RAG Settings
        st.subheader("🎛️ RAG Settings")
        top_k = st.slider("Documents to retrieve", 1, 10, 5)
        use_reranking = st.checkbox("Use reranking", value=True)
        include_verification = st.checkbox("Include verification", value=True)
        
        # Generation Settings
        st.subheader("⚙️ Generation Settings")
        max_tokens = st.slider("Max tokens", 100, 1000, 500)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        This system implements a **Hybrid RAG** approach combining:
        - **BM25** (sparse retrieval)
        - **BGE-M3** (dense embeddings)  
        - **BGE Reranker** (result refinement)
        - **Answer verification** (hallucination detection)
        """)

    # Main interface
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🔍 Document Search", "📊 Evaluation"])
    
    with tab1:
        st.header("Ask Questions")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "verification" in message:
                    st.markdown(message["content"])
                    
                    # Show verification results
                    if message["verification"]:
                        with st.expander("🔍 Verification Details"):
                            display_verification_results(message["verification"])
                    
                    # Show retrieved documents
                    if "documents" in message:
                        with st.expander(f"📄 Retrieved Documents ({len(message['documents'])})"):
                            for i, doc in enumerate(message["documents"], 1):
                                st.markdown(f"**Document {i}** (Score: {doc.get('fusion_score', 0):.3f})")
                                st.markdown(f"*Source: {doc['metadata'].get('title', 'Unknown')}*")
                                st.markdown(f"{doc['document'][:300]}...")
                                st.markdown("---")
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing your question..."):
                    result = complete_rag_pipeline(
                        query=prompt,
                        top_k=top_k,
                        use_reranking=use_reranking,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        include_verification=include_verification
                    )
                
                if result:
                    answer = result["answer"]
                    st.markdown(answer)
                    
                    # Show verification
                    if result.get("verification"):
                        with st.expander("🔍 Verification Details"):
                            display_verification_results(result["verification"])
                    
                    # Show documents
                    if result.get("retrieved_documents"):
                        with st.expander(f"📄 Retrieved Documents ({len(result['retrieved_documents'])})"):
                            for i, doc in enumerate(result["retrieved_documents"], 1):
                                st.markdown(f"**Document {i}** (Fusion Score: {doc.get('fusion_score', 0):.3f})")
                                st.markdown(f"*Source: {doc['metadata'].get('title', 'Unknown')}*")
                                st.markdown(f"{doc['document'][:300]}...")
                                st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "verification": result.get("verification"),
                        "documents": result.get("retrieved_documents", [])
                    })
                else:
                    st.error("Failed to generate response")
    
    with tab2:
        st.header("Document Search")
        st.markdown("Search the knowledge base without generating answers")
        
        search_query = st.text_input("Search query:", placeholder="Enter your search terms...")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            search_top_k = st.selectbox("Number of results", [3, 5, 10, 15], index=1)
        with col2:
            search_reranking = st.checkbox("Use reranking", value=True, key="search_rerank")
        
        if st.button("🔍 Search", type="primary"):
            if search_query:
                with st.spinner("Searching..."):
                    search_results = search_documents(search_query, search_top_k, search_reranking)
                
                if search_results:
                    st.success(f"Found {search_results['total_found']} relevant documents")
                    
                    for i, doc in enumerate(search_results["results"], 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="document-card">
                                <h4>📄 Document {i}</h4>
                                <p><strong>Source:</strong> {doc['metadata'].get('title', 'Unknown')}</p>
                                <p><strong>Fusion Score:</strong> {doc.get('fusion_score', 0):.4f}</p>
                                <p><strong>BM25:</strong> {doc.get('norm_bm25_score', 0):.4f} | 
                                   <strong>Dense:</strong> {doc.get('norm_dense_score', 0):.4f}</p>
                                <hr>
                                <p>{doc['document']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Search failed")
            else:
                st.warning("Please enter a search query")
    
    with tab3:
        st.header("Answer Evaluation")
        st.markdown("Evaluate the quality and factual consistency of generated answers")
        
        # Quick evaluation examples
        if st.button("📝 Load Example Evaluation"):
            st.session_state.eval_query = "What is the University of Notre Dame?"
            st.session_state.eval_answer = "The University of Notre Dame is a Catholic research university located in Indiana. It was founded in 1842 and is famous for its football team."
        
        eval_query = st.text_area("Query:", 
                                value=st.session_state.get("eval_query", ""),
                                placeholder="Enter the original question...")
        
        eval_answer = st.text_area("Generated Answer:", 
                                 value=st.session_state.get("eval_answer", ""),
                                 placeholder="Enter the answer to evaluate...")
        
        if st.button("🔎 Evaluate Answer", type="primary"):
            if eval_query and eval_answer:
                with st.spinner("Evaluating answer..."):
                    # First get relevant documents
                    search_results = search_documents(eval_query, 5, True)
                    
                    if search_results:
                        # Run evaluation
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/evaluate",
                                json={
                                    "query": eval_query,
                                    "generated_answer": eval_answer,
                                    "retrieved_documents": search_results["results"]
                                }
                            )
                            
                            if response.status_code == 200:
                                eval_result = response.json()
                                
                                # Display results
                                st.success("✅ Evaluation completed")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Consistency Score", 
                                        f"{eval_result['evaluation']['consistency_score']:.3f}",
                                        help="0.0 = Inconsistent, 1.0 = Fully consistent"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Confidence Level",
                                        eval_result["confidence_level"],
                                        help="Overall confidence in the answer"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Hallucination Risk",
                                        eval_result['evaluation']['hallucination_risk'],
                                        help="Risk of containing hallucinated information"
                                    )
                                
                                # Detailed verification
                                st.subheader("🔍 Detailed Analysis")
                                display_verification_results(eval_result["evaluation"])
                                
                                # Claim analysis
                                if eval_result["evaluation"].get("claim_verifications"):
                                    st.subheader("📋 Claim-by-Claim Analysis")
                                    
                                    claims_df = pd.DataFrame([
                                        {
                                            "Claim": claim["claim"],
                                            "Verdict": claim["verdict"],
                                            "Explanation": claim["explanation"]
                                        }
                                        for claim in eval_result["evaluation"]["claim_verifications"]
                                    ])
                                    
                                    st.dataframe(claims_df, use_container_width=True)
                            else:
                                st.error(f"Evaluation failed: HTTP {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"Evaluation error: {str(e)}")
                    else:
                        st.error("Failed to retrieve documents for evaluation")
            else:
                st.warning("Please provide both query and answer")
        
        # Clear button
        if st.button("🗑️ Clear"):
            st.session_state.eval_query = ""
            st.session_state.eval_answer = ""
            st.rerun()

if __name__ == "__main__":
    main()