"""
Evaluation Data Collector for Hybrid RAG System
Collects real performance data from your running system
"""
import requests
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import asyncio
import aiohttp

class RAGEvaluationCollector:
    def __init__(self, backend_url="http://localhost:8000", results_dir="evaluation_results"):
        self.backend_url = backend_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test queries for different categories
        self.test_queries = {
            'factual': [
                "Where was the 2008 Olympics held?",
                "What is the University of Notre Dame?",
                "Who won the Nobel Prize in Physics in 2008?",
                "What is the capital of France?",
                "When did World War II end?"
            ],
            'analytical': [
                "How does photosynthesis work?",
                "What causes earthquakes?",
                "Why do leaves change color in fall?",
                "How do computers process information?",
                "What is the greenhouse effect?"
            ],
            'comparative': [
                "What is the difference between BM25 and dense retrieval?",
                "Compare Olympic games in different years",
                "How do different search algorithms work?",
                "Compare renewable vs non-renewable energy",
                "What are the pros and cons of different ML models?"
            ],
            'causal': [
                "What caused the 2008 financial crisis?",
                "Why did the earthquake in Sichuan happen?",
                "What led to the development of the internet?",
                "How did climate change affect weather patterns?",
                "What factors influence economic growth?"
            ]
        }
    
    def check_system_health(self):
        """Check if the RAG system is running"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ System is healthy:")
                for component, status in health_data.items():
                    if isinstance(status, dict):
                        for sub_comp, sub_status in status.items():
                            print(f"   {sub_comp}: {sub_status}")
                    else:
                        print(f"   {component}: {status}")
                return True
            else:
                print(f"❌ System health check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def evaluate_single_query(self, query: str, query_type: str = "general"):
        """Evaluate a single query and collect metrics"""
        try:
            # Start timing
            start_time = time.time()
            
            # Make RAG request
            payload = {
                "query": query,
                "top_k": 5,
                "include_verification": True,
                "use_reranking": True
            }
            
            response = requests.post(f"{self.backend_url}/rag", json=payload, timeout=30)
            
            # End timing
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract metrics
                result = {
                    'query': query,
                    'query_type': query_type,
                    'response_time': response_time,
                    'answer': data.get('answer', ''),
                    'documents_retrieved': len(data.get('retrieved_documents', [])),
                    'verification': data.get('verification', {}),
                    'metadata': data.get('metadata', {}),
                    'timestamp': time.time()
                }
                
                # Process retrieval scores
                if 'retrieved_documents' in data:
                    docs = data['retrieved_documents']
                    result['bm25_scores'] = [doc.get('bm25_score', 0) for doc in docs]
                    result['dense_scores'] = [doc.get('dense_score', 0) for doc in docs]
                    result['fusion_scores'] = [doc.get('fusion_score', 0) for doc in docs]
                    result['rerank_scores'] = [doc.get('rerank_score', 0) for doc in docs]
                
                # Process verification results
                if 'verification' in data:
                    verification = data['verification']
                    result['overall_verdict'] = verification.get('overall_verdict', 'UNKNOWN')
                    result['consistency_score'] = verification.get('consistency_score', 0)
                    result['hallucination_risk'] = verification.get('hallucination_risk', 'UNKNOWN')
                
                return result
            else:
                print(f"❌ Query failed: {query} - HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Query error: {query} - {e}")
            return None
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all query types"""
        print("🚀 Starting comprehensive RAG system evaluation...")
        print("="*60)
        
        # Check system health first
        if not self.check_system_health():
            print("❌ System not ready. Please start the RAG system first.")
            return None
        
        all_results = []
        
        # Evaluate each query type
        for query_type, queries in self.test_queries.items():
            print(f"\n📊 Evaluating {query_type.upper()} queries...")
            
            for i, query in enumerate(queries, 1):
                print(f"   {i}/{len(queries)}: {query[:50]}...")
                result = self.evaluate_single_query(query, query_type)
                
                if result:
                    all_results.append(result)
                    print(f"      ✅ Response time: {result['response_time']:.2f}s")
                    print(f"      📄 Documents: {result['documents_retrieved']}")
                    print(f"      🔍 Verdict: {result['overall_verdict']}")
                else:
                    print(f"      ❌ Failed")
                
                # Small delay to avoid overwhelming the system
                time.sleep(1)
        
        # Save raw results
        results_file = self.results_dir / "raw_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📁 Raw results saved to: {results_file}")
        
        # Process and aggregate results
        processed_results = self.process_evaluation_results(all_results)
        
        # Save processed results
        processed_file = self.results_dir / "processed_evaluation_results.json"
        with open(processed_file, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        print(f"📊 Processed results saved to: {processed_file}")
        
        return processed_results
    
    def process_evaluation_results(self, raw_results: List[Dict]):
        """Process raw results into plotting-ready format"""
        if not raw_results:
            return {}
        
        df = pd.DataFrame(raw_results)
        
        processed = {
            'summary_stats': {
                'total_queries': len(df),
                'avg_response_time': df['response_time'].mean(),
                'median_response_time': df['response_time'].median(),
                'avg_documents_retrieved': df['documents_retrieved'].mean(),
                'success_rate': len(df) / (len(df) + 0),  # Assuming all successful made it here
            },
            'query_type_performance': {},
            'retrieval_analysis': {},
            'verification_analysis': {},
            'response_time_analysis': {}
        }
        
        # Query type performance
        for query_type in df['query_type'].unique():
            type_data = df[df['query_type'] == query_type]
            processed['query_type_performance'][query_type] = {
                'avg_response_time': type_data['response_time'].mean(),
                'avg_consistency_score': type_data['consistency_score'].mean(),
                'verdict_distribution': type_data['overall_verdict'].value_counts().to_dict()
            }
        
        # Retrieval analysis
        all_bm25_scores = []
        all_dense_scores = []
        all_fusion_scores = []
        all_rerank_scores = []
        
        for result in raw_results:
            all_bm25_scores.extend(result.get('bm25_scores', []))
            all_dense_scores.extend(result.get('dense_scores', []))
            all_fusion_scores.extend(result.get('fusion_scores', []))
            all_rerank_scores.extend(result.get('rerank_scores', []))
        
        processed['retrieval_analysis'] = {
            'bm25_scores': all_bm25_scores,
            'dense_scores': all_dense_scores,
            'fusion_scores': all_fusion_scores,
            'rerank_scores': all_rerank_scores,
            'score_statistics': {
                'bm25_mean': np.mean(all_bm25_scores) if all_bm25_scores else 0,
                'dense_mean': np.mean(all_dense_scores) if all_dense_scores else 0,
                'fusion_mean': np.mean(all_fusion_scores) if all_fusion_scores else 0,
                'rerank_mean': np.mean(all_rerank_scores) if all_rerank_scores else 0,
            }
        }
        
        # Verification analysis
        processed['verification_analysis'] = {
            'overall_verdict_distribution': df['overall_verdict'].value_counts().to_dict(),
            'hallucination_risk_distribution': df['hallucination_risk'].value_counts().to_dict(),
            'consistency_scores': df['consistency_score'].tolist(),
            'avg_consistency_score': df['consistency_score'].mean()
        }
        
        # Response time analysis
        processed['response_time_analysis'] = {
            'by_query_type': df.groupby('query_type')['response_time'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            'overall_distribution': df['response_time'].tolist(),
            'percentiles': {
                'p50': df['response_time'].quantile(0.5),
                'p90': df['response_time'].quantile(0.9),
                'p95': df['response_time'].quantile(0.95),
                'p99': df['response_time'].quantile(0.99)
            }
        }
        
        return processed
    
    def generate_comparison_data(self):
        """Generate data comparing different retrieval methods"""
        print("🔄 Generating method comparison data...")
        
        comparison_queries = [
            "Where was the 2008 Olympics held?",
            "What caused the 2008 Sichuan earthquake?",
            "How does the Olympic torch relay work?",
        ]
        
        methods_data = {
            'bm25_only': [],
            'dense_only': [],
            'hybrid': [],
            'hybrid_with_rerank': []
        }
        
        for query in comparison_queries:
            print(f"   Testing: {query}")
            
            # Test different configurations
            configs = [
                {'use_reranking': False, 'fusion_alpha': 1.0},  # BM25 only
                {'use_reranking': False, 'fusion_alpha': 0.0},  # Dense only  
                {'use_reranking': False, 'fusion_alpha': 0.3},  # Hybrid
                {'use_reranking': True, 'fusion_alpha': 0.3}    # Hybrid + rerank
            ]
            
            for i, config in enumerate(configs):
                try:
                    payload = {
                        "query": query,
                        "top_k": 5,
                        "include_verification": False,
                        **config
                    }
                    
                    start_time = time.time()
                    response = requests.post(f"{self.backend_url}/search", json=payload, timeout=15)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        method_key = list(methods_data.keys())[i]
                        
                        result = {
                            'query': query,
                            'response_time': end_time - start_time,
                            'num_results': len(data.get('results', [])),
                            'scores': [doc.get('fusion_score', 0) for doc in data.get('results', [])]
                        }
                        
                        methods_data[method_key].append(result)
                    
                except Exception as e:
                    print(f"   ❌ Error with config {i}: {e}")
                
                time.sleep(0.5)
        
        return methods_data

if __name__ == "__main__":
    collector = RAGEvaluationCollector()
    
    print("🔬 RAG System Evaluation Data Collector")
    print("="*50)
    
    # Run comprehensive evaluation
    results = collector.run_comprehensive_evaluation()
    
    if results:
        print("\n📊 EVALUATION SUMMARY:")
        print("-"*30)
        stats = results['summary_stats']
        print(f"Total Queries Evaluated: {stats['total_queries']}")
        print(f"Average Response Time: {stats['avg_response_time']:.2f}s")
        print(f"Median Response Time: {stats['median_response_time']:.2f}s")
        print(f"Average Documents Retrieved: {stats['avg_documents_retrieved']:.1f}")
        
        # Generate comparison data
        comparison_data = collector.generate_comparison_data()
        
        # Save comparison data
        comparison_file = collector.results_dir / "method_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n✅ All evaluation data collected successfully!")
        print(f"📁 Results directory: {collector.results_dir}")
        print("\n🎯 Ready for plotting! Run plot_results.py next.")
    else:
        print("\n❌ Evaluation failed. Please check your system and try again.")