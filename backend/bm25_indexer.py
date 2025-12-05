"""
BM25 Indexer for sparse retrieval
"""
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path

class BM25Indexer:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
    def load_chunks(self, chunks_path: str) -> List[Dict]:
        """Load processed chunks"""
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks for BM25 indexing")
        return chunks
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 (tokenization)"""
        # Simple tokenization - split on whitespace and convert to lowercase
        tokens = text.lower().split()
        # Remove punctuation and keep only alphanumeric
        tokens = [''.join(c for c in token if c.isalnum()) for token in tokens]
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        return tokens
    
    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from chunks"""
        print("Building BM25 index...")
        
        # Prepare documents and metadata
        tokenized_docs = []
        self.documents = []
        self.doc_metadata = []
        
        for chunk in chunks:
            text = chunk['text']
            tokens = self.preprocess_text(text)
            
            tokenized_docs.append(tokens)
            self.documents.append(text)
            self.doc_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'title': chunk['title'],
                'related_questions': chunk.get('related_questions', []),
                'related_answers': chunk.get('related_answers', [])
            })
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 index built with {len(tokenized_docs)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search using BM25"""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self.preprocess_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((
                    self.documents[idx],  # document text
                    float(scores[idx]),   # BM25 score
                    self.doc_metadata[idx]  # metadata
                ))
        
        return results
    
    def save_index(self, output_path: str):
        """Save BM25 index and metadata"""
        index_data = {
            'bm25': self.bm25,
            'documents': self.documents,
            'doc_metadata': self.doc_metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"BM25 index saved to {output_path}")
    
    def load_index(self, input_path: str):
        """Load BM25 index and metadata"""
        with open(input_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.documents = index_data['documents']
        self.doc_metadata = index_data['doc_metadata']
        print(f"BM25 index loaded from {input_path}")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the BM25 index"""
        if self.bm25 is None:
            return {}
        
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.bm25.idf),
            'avg_doc_length': np.mean([len(doc.split()) for doc in self.documents])
        }

def main():
    """Build and save BM25 index"""
    indexer = BM25Indexer()
    
    # Load processed chunks
    chunks = indexer.load_chunks("../data/processed_chunks.pkl")
    
    # Build index
    indexer.build_index(chunks)
    
    # Save index
    output_dir = Path("../data")
    indexer.save_index(str(output_dir / "bm25_index.pkl"))
    
    # Print stats
    stats = indexer.get_index_stats()
    print("\nBM25 Index Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test search
    print("\nTesting BM25 search:")
    test_query = "Notre Dame university"
    results = indexer.search(test_query, top_k=3)
    
    print(f"\nTop 3 results for '{test_query}':")
    for i, (doc, score, metadata) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"Title: {metadata['title']}")
        print(f"Text: {doc[:200]}...")

if __name__ == "__main__":
    main()