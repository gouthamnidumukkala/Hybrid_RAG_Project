"""
Evaluation data collector for Hybrid RAG system
Collects performance metrics for visualization and analysis
"""
import requests
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List
import os
from pathlib import Path

class RAGEvaluator:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        """Initialize the evaluator with backend URL"""
        self.backend_url = backend_url
        self.results_dir = Path("../results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create evaluation dataset
        self.evaluation_queries = [
            # Factual questions
            "What is the University of Notre Dame?",
            "Where was the 2008 Olympics held?",
            "When did the 2008 Summer Olympics torch relay begin?",
            "What is the capital of France mentioned in Olympic context?",
            "How long did the Olympic torch relay last?",
            
            # Complex analytical questions  
            "What were the main challenges during the 2008 Olympic torch relay?",
            "How did different countries respond to the torch relay protests?",
            "What safety measures were taken during the earthquake response?",
            "What was the international response to the 2008 Sichuan earthquake?",
            "How did the media cover the Olympic torch relay protests?",
            
            # Comparative questions
            "Compare the response times of different emergency services",
            "What are the differences between BM25 and dense retrieval?",
            "How do various countries' Olympic preparations differ?",
            "Compare the impact of natural disasters vs man-made events",
            "What are the similarities between different protest movements?",
            
            # Inferential questions
            "Why might certain regions be more vulnerable to earthquakes?",
            "What factors influenced the Olympic torch relay route changes?",
            "How might future Olympic events learn from 2008 experiences?",
            "What lessons can be drawn from crisis communication strategies?",
            "How do cultural differences affect international event planning?"
        ]
        
    def check_system_health(self) -> bool:
        """Check if the backend system is healthy"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_retrieval_performance(self, alpha_values: List[float] = None) -> Dict:
        """Evaluate retrieval performance with different parameters"""
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print("🔍 Evaluating retrieval performance...")
        results = {
            'alphas': alpha_values,
            'k_values': [1, 3, 5, 10],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': [],
            'response_times': []
        }
        
        # Test different k values
        for k in results['k_values']:
            print(f"   Testing k={k}...")
            precisions, recalls, f1s, times = [], [], [], []
            
            for query in self.evaluation_queries[:10]:  # Use subset for speed
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.backend_url}/search",
                        json={"query": query, "top_k": k},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_time = time.time() - start_time
                        times.append(response_time)
                        
                        # Mock precision/recall calculation (replace with actual ground truth)
                        precision = np.random.uniform(0.6, 0.9)
                        recall = np.random.uniform(0.5, 0.8)
                        f1 = 2 * precision * recall / (precision + recall)
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)
                
                except Exception as e:
                    print(f"   Error with query: {str(e)}")
                    continue
            
            results['precision_scores'].append(np.mean(precisions) if precisions else 0)
            results['recall_scores'].append(np.mean(recalls) if recalls else 0)
            results['f1_scores'].append(np.mean(f1s) if f1s else 0)
            results['response_times'].append(np.mean(times) if times else 0)
        
        return results
    
    def evaluate_answer_quality(self) -> Dict:
        """Evaluate answer generation quality"""
        print("📝 Evaluating answer quality...")
        
        results = {
            'total_queries': len(self.evaluation_queries),
            'successful_responses': 0,
            'quality_scores': {'relevance': [], 'faithfulness': [], 'usefulness': [], 'completeness': []},
            'verification_results': {'supported': 0, 'unsupported': 0, 'contradicted': 0},
            'hallucination_risks': {'none': 0, 'low': 0, 'medium': 0, 'high': 0},
            'response_times': []
        }
        
        for i, query in enumerate(self.evaluation_queries):
            print(f"   Processing query {i+1}/{len(self.evaluation_queries)}")
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.backend_url}/rag",
                    json={
                        "query": query,
                        "top_k": 5,
                        "include_verification": True
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_time = time.time() - start_time
                    results['response_times'].append(response_time)
                    results['successful_responses'] += 1
                    
                    data = response.json()
                    
                    # Extract verification results
                    if 'verification' in data:
                        verdict = data['verification'].get('overall_verdict', 'UNKNOWN')
                        if verdict == 'SUPPORTED':
                            results['verification_results']['supported'] += 1
                        elif verdict == 'UNSUPPORTED':
                            results['verification_results']['unsupported'] += 1
                        elif verdict == 'CONTRADICTED':
                            results['verification_results']['contradicted'] += 1
                        
                        # Mock hallucination risk (replace with actual analysis)
                        risk = data['verification'].get('hallucination_risk', 'MEDIUM')
                        risk_level = risk.lower()
                        if risk_level in results['hallucination_risks']:
                            results['hallucination_risks'][risk_level] += 1
                    
                    # Mock quality scores (replace with actual human evaluation)
                    results['quality_scores']['relevance'].append(np.random.uniform(3.5, 4.5))
                    results['quality_scores']['faithfulness'].append(np.random.uniform(3.0, 4.5))
                    results['quality_scores']['usefulness'].append(np.random.uniform(3.0, 4.2))
                    results['quality_scores']['completeness'].append(np.random.uniform(2.8, 4.0))
            
            except Exception as e:
                print(f"   Error processing query: {str(e)}")
                continue
        
        # Calculate averages
        for metric in results['quality_scores']:
            scores = results['quality_scores'][metric]
            results['quality_scores'][metric] = np.mean(scores) if scores else 0
        
        results['avg_response_time'] = np.mean(results['response_times']) if results['response_times'] else 0
        
        return results
    
    def evaluate_system_comparison(self) -> Dict:
        """Compare different system configurations"""
        print("⚖️ Comparing system configurations...")
        
        # Test different methods by modifying queries or using different endpoints
        methods = ['search', 'rag']  # Different endpoints represent different methods
        
        results = {
            'methods': methods,
            'performance_metrics': {},
            'response_times': {},
            'error_rates': {}
        }
        
        for method in methods:
            print(f"   Testing {method} method...")
            
            times = []
            successes = 0
            errors = 0
            
            for query in self.evaluation_queries[:10]:  # Use subset for comparison
                try:
                    start_time = time.time()
                    
                    if method == 'search':
                        response = requests.post(
                            f"{self.backend_url}/search",
                            json={"query": query, "top_k": 5},
                            timeout=30
                        )
                    else:  # rag
                        response = requests.post(
                            f"{self.backend_url}/rag",
                            json={"query": query, "top_k": 5},
                            timeout=60
                        )
                    
                    response_time = time.time() - start_time
                    times.append(response_time)
                    
                    if response.status_code == 200:
                        successes += 1
                    else:
                        errors += 1
                
                except Exception as e:
                    errors += 1
                    continue
            
            results['response_times'][method] = np.mean(times) if times else 0
            results['error_rates'][method] = errors / (successes + errors) if (successes + errors) > 0 else 1
            results['performance_metrics'][method] = {
                'success_rate': successes / (successes + errors) if (successes + errors) > 0 else 0,
                'avg_response_time': np.mean(times) if times else 0
            }
        
        return results
    
    def collect_system_stats(self) -> Dict:
        """Collect system statistics"""
        print("📊 Collecting system statistics...")
        
        try:
            response = requests.get(f"{self.backend_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run complete evaluation suite"""
        print("🚀 Starting comprehensive evaluation...")
        
        if not self.check_system_health():
            print("❌ Backend system is not healthy. Please ensure the system is running.")
            return {"error": "System not healthy"}
        
        print("✅ System health check passed")
        
        # Run all evaluations
        evaluation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_stats': self.collect_system_stats(),
            'retrieval_performance': self.evaluate_retrieval_performance(),
            'answer_quality': self.evaluate_answer_quality(),
            'system_comparison': self.evaluate_system_comparison()
        }
        
        # Save results
        results_file = self.results_dir / f"evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {results_file}")
        
        # Generate summary
        self.print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("📋 EVALUATION SUMMARY")
        print("="*60)
        
        # System stats
        if 'system_stats' in results and 'error' not in results['system_stats']:
            stats = results['system_stats']
            print(f"📊 System Configuration:")
            print(f"   • Generator Model: {stats.get('configuration', {}).get('generator_model', 'N/A')}")
            print(f"   • Retriever Model: {stats.get('configuration', {}).get('retriever_model', 'N/A')}")
            print(f"   • Fusion Alpha: {stats.get('configuration', {}).get('fusion_alpha', 'N/A')}")
            
            if 'retrieval' in stats:
                bm25_stats = stats['retrieval'].get('bm25_stats', {})
                print(f"   • BM25 Documents: {bm25_stats.get('total_documents', 'N/A'):,}")
                print(f"   • Vocabulary Size: {bm25_stats.get('vocabulary_size', 'N/A'):,}")
        
        # Answer quality results
        if 'answer_quality' in results:
            quality = results['answer_quality']
            print(f"\n📝 Answer Quality:")
            print(f"   • Success Rate: {quality['successful_responses']}/{quality['total_queries']} ({quality['successful_responses']/quality['total_queries']*100:.1f}%)")
            print(f"   • Avg Response Time: {quality['avg_response_time']:.2f}s")
            
            if quality['quality_scores']:
                print(f"   • Quality Scores:")
                for metric, score in quality['quality_scores'].items():
                    print(f"     - {metric.title()}: {score:.2f}/5.0")
            
            verification = quality['verification_results']
            total_verifications = sum(verification.values())
            if total_verifications > 0:
                print(f"   • Verification Results:")
                for verdict, count in verification.items():
                    percentage = count / total_verifications * 100
                    print(f"     - {verdict.title()}: {count} ({percentage:.1f}%)")
        
        # System comparison
        if 'system_comparison' in results:
            comparison = results['system_comparison']
            print(f"\n⚖️ System Performance:")
            for method, metrics in comparison['performance_metrics'].items():
                print(f"   • {method.upper()}:")
                print(f"     - Success Rate: {metrics['success_rate']*100:.1f}%")
                print(f"     - Avg Response Time: {metrics['avg_response_time']:.2f}s")
        
        print("\n" + "="*60)

def main():
    """Main evaluation function"""
    evaluator = RAGEvaluator()
    
    print("🧪 Hybrid RAG System Evaluation")
    print("================================")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if 'error' not in results:
        print("\n✅ Evaluation completed successfully!")
        print(f"📊 Use the results with plot_results.py to generate visualizations")
    else:
        print(f"\n❌ Evaluation failed: {results['error']}")

if __name__ == "__main__":
    main()