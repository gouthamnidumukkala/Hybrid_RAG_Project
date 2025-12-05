"""
Data processor for SQuAD dataset
Handles loading, chunking, and preprocessing of data for RAG system
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import pickle
from pathlib import Path

class SQuADDataProcessor:
    def __init__(self, data_path: str, chunk_size: int = 512, chunk_overlap: int = 50):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data = None
        self.chunks = []
        
    def load_data(self) -> pd.DataFrame:
        """Load SQuAD CSV data"""
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} rows")
        return self.data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()]', '', text)
        return text.strip()
    
    def create_chunks(self, text: str, doc_id: str) -> List[Dict]:
        """Create overlapping chunks from text"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{len(chunks)}",
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + self.chunk_size, len(words))
            })
            
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
    
    def process_dataset(self) -> List[Dict]:
        """Process SQuAD dataset into chunks"""
        if self.data is None:
            self.load_data()
        
        print("Processing dataset into chunks...")
        all_chunks = []
        
        # Group by title to create document chunks
        grouped = self.data.groupby('title')
        
        for title, group in grouped:
            # Combine all contexts for this title
            contexts = group['context'].unique()
            
            for i, context in enumerate(contexts):
                cleaned_context = self.clean_text(context)
                if len(cleaned_context) < 50:  # Skip very short contexts
                    continue
                    
                doc_id = f"{title}_{i}"
                chunks = self.create_chunks(cleaned_context, doc_id)
                
                # Add metadata
                for chunk in chunks:
                    chunk.update({
                        'title': title,
                        'original_context': context,
                        'related_questions': group[group['context'] == context]['question'].tolist(),
                        'related_answers': group[group['context'] == context]['answer'].tolist()
                    })
                
                all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks")
        return all_chunks
    
    def save_chunks(self, output_path: str):
        """Save processed chunks to pickle file"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Saved chunks to {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Dict]:
        """Load processed chunks from pickle file"""
        with open(input_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunks from {input_path}")
        return self.chunks
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the processed dataset"""
        if not self.chunks:
            return {}
        
        chunk_lengths = [len(chunk['text'].split()) for chunk in self.chunks]
        unique_titles = set(chunk['title'] for chunk in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'unique_documents': len(unique_titles),
            'avg_chunk_length': np.mean(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_words': sum(chunk_lengths)
        }

def main():
    """Process SQuAD dataset and save chunks"""
    processor = SQuADDataProcessor(
        data_path="../data/SQuAD-v1.1.csv",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Process data
    chunks = processor.process_dataset()
    
    # Save processed chunks
    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)
    processor.save_chunks(output_dir / "processed_chunks.pkl")
    
    # Print stats
    stats = processor.get_document_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()