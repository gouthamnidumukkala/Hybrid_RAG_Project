"""
Hybrid Retrieval System
Combines BM25 (sparse) and Dense (ChromaDB) retrieval with weighted fusion
"""
import numpy as np
from typing import List, Dict, Tuple, Set
from bm25_indexer import BM25Indexer
from chroma_store import ChromaVectorStore
import requests

class HybridRetriever:
    def __init__(self, 
                 bm25_index_path: str, 
                 chroma_db_path: str,
                 fusion_alpha: float = 0.3,
                 reranker_model: str = "xitao/bge-reranker-v2-m3"):
        """
        Initialize hybrid retriever
        
        Args:
            bm25_index_path: Path to BM25 index
            chroma_db_path: Path to ChromaDB
            fusion_alpha: Weight for BM25 vs dense (α * BM25 + (1-α) * dense)
            reranker_model: Ollama reranker model name
        """
        self.fusion_alpha = fusion_alpha
        self.reranker_model = reranker_model
        self.ollama_base_url = "http://localhost:11434"
        
        # Initialize retrievers
        self.bm25_retriever = BM25Indexer()
        self.dense_retriever = ChromaVectorStore(db_path=chroma_db_path)
        
        # Load indices
        self.load_indices(bm25_index_path)
        
    def load_indices(self, bm25_index_path: str):
        """Load BM25 and ChromaDB indices"""
        print("Loading BM25 index...")
        self.bm25_retriever.load_index(bm25_index_path)
        
        print("Initializing ChromaDB...")
        self.dense_retriever.initialize_client()
        self.dense_retriever.create_collection()
        
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization"""
        if not scores:
            return scores
            
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
            
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def get_bm25_results(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Get BM25 results"""
        return self.bm25_retriever.search(query, top_k)
    
    def get_dense_results(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Get dense retrieval results"""
        return self.dense_retriever.search(query, top_k)
    
    def merge_and_deduplicate(self, 
                             bm25_results: List[Tuple[str, float, Dict]], 
                             dense_results: List[Tuple[str, float, Dict]]) -> List[Dict]:
        """Merge results and remove duplicates based on text content"""
        seen_texts: Set[str] = set()
        merged_results = []
        
        # Create lookup for dense results
        dense_lookup = {doc: (score, metadata) for doc, score, metadata in dense_results}
        
        # Process BM25 results first
        for doc, bm25_score, metadata in bm25_results:
            if doc not in seen_texts:
                dense_score = 0.0
                if doc in dense_lookup:
                    dense_score, _ = dense_lookup[doc]
                
                merged_results.append({
                    'document': doc,
                    'metadata': metadata,
                    'bm25_score': bm25_score,
                    'dense_score': dense_score,
                    'source': 'bm25'
                })
                seen_texts.add(doc)
        
        # Add remaining dense results
        for doc, dense_score, metadata in dense_results:
            if doc not in seen_texts:
                merged_results.append({
                    'document': doc,
                    'metadata': metadata,
                    'bm25_score': 0.0,
                    'dense_score': dense_score,
                    'source': 'dense'
                })
                seen_texts.add(doc)
        
        return merged_results
    
    def apply_fusion(self, merged_results: List[Dict]) -> List[Dict]:
        """Apply weighted fusion to combined results"""
        # Extract scores
        bm25_scores = [result['bm25_score'] for result in merged_results]
        dense_scores = [result['dense_score'] for result in merged_results]
        
        # Normalize scores
        norm_bm25 = self.normalize_scores(bm25_scores)
        norm_dense = self.normalize_scores(dense_scores)
        
        # Apply fusion: α * BM25 + (1-α) * dense
        for i, result in enumerate(merged_results):
            fusion_score = (self.fusion_alpha * norm_bm25[i] + 
                          (1 - self.fusion_alpha) * norm_dense[i])
            result['fusion_score'] = fusion_score
            result['norm_bm25_score'] = norm_bm25[i]
            result['norm_dense_score'] = norm_dense[i]
        
        # Sort by fusion score
        merged_results.sort(key=lambda x: x['fusion_score'], reverse=True)
        
        return merged_results
    
    def rerank_with_ollama(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rerank documents using Ollama reranker model"""
        if not documents:
            return []
        
        try:
            # Prepare reranking request
            rerank_pairs = []
            for doc in documents:
                # Create query-document pairs for reranking
                rerank_pairs.append(f"Query: {query}\nDocument: {doc}")
            
            # For now, use a simple approach - get embeddings and compute similarity
            # In a full implementation, you'd use a proper reranker
            query_embedding = self.dense_retriever.get_embeddings_from_ollama([query])
            doc_embeddings = self.dense_retriever.get_embeddings_from_ollama(documents)
            
            if not query_embedding or not doc_embeddings:
                return [(doc, 1.0) for doc in documents[:top_k]]
            
            # Compute cosine similarity
            query_vec = np.array(query_embedding[0])
            similarities = []
            
            for doc_emb in doc_embeddings:
                if doc_emb:
                    doc_vec = np.array(doc_emb)
                    similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    similarities.append(float(similarity))
                else:
                    similarities.append(0.0)
            
            # Sort by similarity
            doc_sim_pairs = list(zip(documents, similarities))
            doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return doc_sim_pairs[:top_k]
            
        except Exception as e:
            print(f"Reranking failed: {str(e)}")
            # Fallback to original order
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    def hybrid_search(self, 
                     query: str, 
                     top_k_bm25: int = 10, 
                     top_k_dense: int = 10,
                     final_top_k: int = 5,
                     use_reranking: bool = True) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and dense retrieval
        
        Args:
            query: Search query
            top_k_bm25: Number of results from BM25
            top_k_dense: Number of results from dense retrieval
            final_top_k: Final number of results to return
            use_reranking: Whether to apply reranking
            
        Returns:
            List of ranked results with scores and metadata
        """
        print(f"Hybrid search for: '{query}'")
        
        # Get results from both retrievers
        print("Getting BM25 results...")
        bm25_results = self.get_bm25_results(query, top_k_bm25)
        
        print("Getting dense results...")
        dense_results = self.get_dense_results(query, top_k_dense)
        
        # Merge and deduplicate
        print("Merging results...")
        merged_results = self.merge_and_deduplicate(bm25_results, dense_results)
        
        # Apply fusion
        print("Applying fusion...")
        fused_results = self.apply_fusion(merged_results)
        
        # Take top results before reranking
        top_results = fused_results[:final_top_k * 2]  # Get more for reranking
        
        if use_reranking and len(top_results) > 1:
            print("Applying reranking...")
            documents = [result['document'] for result in top_results]
            reranked_docs = self.rerank_with_ollama(query, documents, final_top_k)
            
            # Map reranked results back to original format
            rerank_lookup = {doc: score for doc, score in reranked_docs}
            final_results = []
            
            for doc, rerank_score in reranked_docs:
                # Find original result
                for result in top_results:
                    if result['document'] == doc:
                        result['rerank_score'] = rerank_score
                        final_results.append(result)
                        break
                        
                if len(final_results) >= final_top_k:
                    break
        else:
            final_results = top_results[:final_top_k]
            for result in final_results:
                result['rerank_score'] = result['fusion_score']
        
        return final_results
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval system"""
        bm25_stats = self.bm25_retriever.get_index_stats()
        chroma_stats = self.dense_retriever.get_collection_stats()
        
        return {
            'fusion_alpha': self.fusion_alpha,
            'reranker_model': self.reranker_model,
            'bm25_stats': bm25_stats,
            'chroma_stats': chroma_stats
        }

def main():
    """Test hybrid retrieval system"""
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        bm25_index_path="../data/bm25_index.pkl",
        chroma_db_path="../data/chroma_db",
        fusion_alpha=0.3
    )
    
    # Test queries
    test_queries = [
        "Notre Dame university",
        "What is machine learning?",
        "Olympic torch relay"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print('='*60)
        
        results = retriever.hybrid_search(
            query=query,
            top_k_bm25=5,
            top_k_dense=5,
            final_top_k=3,
            use_reranking=True
        )
        
        print(f"\nTop 3 hybrid results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Fusion Score: {result['fusion_score']:.4f}")
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
            print(f"   BM25: {result['norm_bm25_score']:.4f}, Dense: {result['norm_dense_score']:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   Title: {result['metadata'].get('title', 'N/A')}")
            print(f"   Text: {result['document'][:150]}...")
    
    # Print stats
    print(f"\n{'='*60}")
    print("Retrieval System Statistics:")
    print('='*60)
    stats = retriever.get_retrieval_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()