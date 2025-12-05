"""
ChromaDB Vector Store for dense retrieval
Uses Ollama's BGE-M3 model for embeddings
"""
import chromadb
import pickle
import requests
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import time

class ChromaVectorStore:
    def __init__(self, db_path: str = "../data/chroma_db", collection_name: str = "squad_chunks"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.ollama_base_url = "http://localhost:11434"
        
    def initialize_client(self):
        """Initialize ChromaDB client"""
        # Create persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        print(f"ChromaDB client initialized at {self.db_path}")
        
    def get_embeddings_from_ollama(self, texts: List[str], model: str = "bge-m3") -> List[List[float]]:
        """Get embeddings from Ollama BGE-M3 model"""
        embeddings = []
        
        print(f"Getting embeddings for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text
                    }
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                else:
                    print(f"Error getting embedding for text {i}: {response.status_code}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 1024)  # BGE-M3 dimension
                    
            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
                embeddings.append([0.0] * 1024)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")
        
        return embeddings
    
    def create_collection(self):
        """Create or get collection"""
        if self.client is None:
            self.initialize_client()
            
        # Try to get existing collection, create if doesn't exist
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Retrieved existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 50):
        """Add documents to ChromaDB with embeddings"""
        if self.collection is None:
            self.create_collection()
        
        print(f"Adding {len(chunks)} documents to ChromaDB...")
        
        # Process in batches to avoid memory issues
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            texts = [chunk['text'] for chunk in batch]
            ids = [chunk['chunk_id'] for chunk in batch]
            
            # Prepare metadata
            metadatas = []
            for chunk in batch:
                metadata = {
                    'doc_id': chunk['doc_id'],
                    'title': chunk['title'],
                    'chunk_id': chunk['chunk_id'],
                    'start_idx': chunk['start_idx'],
                    'end_idx': chunk['end_idx']
                }
                # Add related questions and answers (limit to avoid metadata size issues)
                if chunk.get('related_questions'):
                    metadata['related_questions'] = str(chunk['related_questions'][:3])  # Limit to 3
                if chunk.get('related_answers'):
                    metadata['related_answers'] = str(chunk['related_answers'][:3])
                metadatas.append(metadata)
            
            # Get embeddings for this batch
            embeddings = self.get_embeddings_from_ollama(texts)
            
            # Add to collection
            try:
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Small delay to avoid overwhelming Ollama
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search similar documents using dense retrieval"""
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        # Get query embedding
        query_embeddings = self.get_embeddings_from_ollama([query])
        
        if not query_embeddings or not query_embeddings[0]:
            return []
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            distances = results['distances'][0] if results['distances'] else [0] * len(documents)
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            
            for doc, distance, metadata in zip(documents, distances, metadatas):
                # Convert distance to similarity score (cosine distance -> cosine similarity)
                similarity = 1.0 - distance
                formatted_results.append((doc, float(similarity), metadata))
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.collection is None:
            return {}
        
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'db_path': self.db_path
        }

def main():
    """Build and populate ChromaDB vector store"""
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Error: Ollama is not running. Please start Ollama first.")
            return
    except:
        print("Error: Cannot connect to Ollama. Please start Ollama first.")
        return
    
    # Initialize vector store
    vector_store = ChromaVectorStore()
    vector_store.initialize_client()
    
    # Load processed chunks
    with open("../data/processed_chunks.pkl", 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Create collection and add documents
    vector_store.create_collection()
    
    # Add documents in smaller batches for testing
    print("Adding documents to ChromaDB (this may take a while)...")
    sample_chunks = chunks[:100]  # Start with 100 chunks for testing
    vector_store.add_documents(sample_chunks, batch_size=10)
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print("\nChromaDB Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test search
    print("\nTesting dense retrieval:")
    test_query = "Notre Dame university"
    results = vector_store.search(test_query, top_k=3)
    
    print(f"\nTop 3 results for '{test_query}':")
    for i, (doc, score, metadata) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {score:.4f}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Text: {doc[:200]}...")

if __name__ == "__main__":
    main()