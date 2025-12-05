"""
Generate Sample Results for Research Plotting
Creates realistic sample data for demonstration purposes
"""
import json
import numpy as np
from datetime import datetime
import os

def generate_sample_results():
    """Generate realistic sample results for plotting"""
    
    # Test queries similar to what would be in SQuAD
    test_queries = [
        "What is the University of Notre Dame?",
        "Where was the 2008 Olympics held?",
        "What happened during the 2008 Sichuan earthquake?",
        "What were the Olympic torch relay protests?",
        "How was the earthquake damage assessed?",
        "What countries participated in the torch relay?",
        "What was the Olympic theme?",
        "How many people were affected by the earthquake?",
        "What were the rescue operations?",
        "How was international aid provided?"
    ]
    
    # Generate realistic results
    results = {
        "queries": [],
        "summary_stats": {},
        "fusion_analysis": [],
        "timestamp": datetime.now().isoformat(),
        "system_info": "Sample data for research plotting"
    }
    
    # Generate query results
    np.random.seed(42)  # For reproducible results
    
    for i, query in enumerate(test_queries):
        # Simulate realistic retrieval scores
        bm25_score = max(0.1, np.random.normal(0.75, 0.15))
        dense_score = max(0.1, np.random.normal(0.72, 0.12))
        hybrid_score = max(bm25_score, dense_score, np.random.normal(0.85, 0.10))
        
        # Response time simulation (some queries are more complex)
        base_time = 2.0
        complexity_factor = len(query.split()) * 0.1
        response_time = base_time + complexity_factor + np.random.normal(0, 0.3)
        response_time = max(1.0, response_time)
        
        # Verification results (better for hybrid)
        if hybrid_score > 0.8:
            verdict = np.random.choice(['SUPPORTED', 'PARTIAL'], p=[0.8, 0.2])
            risk = np.random.choice(['LOW', 'MEDIUM'], p=[0.7, 0.3])
            consistency = np.random.normal(0.85, 0.1)
        elif hybrid_score > 0.6:
            verdict = np.random.choice(['SUPPORTED', 'PARTIAL', 'UNSUPPORTED'], p=[0.5, 0.3, 0.2])
            risk = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.3, 0.5, 0.2])
            consistency = np.random.normal(0.65, 0.15)
        else:
            verdict = np.random.choice(['PARTIAL', 'UNSUPPORTED'], p=[0.3, 0.7])
            risk = np.random.choice(['MEDIUM', 'HIGH'], p=[0.4, 0.6])
            consistency = np.random.normal(0.35, 0.15)
        
        consistency = np.clip(consistency, 0, 1)
        
        query_result = {
            "query": query,
            "bm25_avg_score": round(bm25_score, 3),
            "dense_avg_score": round(dense_score, 3), 
            "hybrid_avg_score": round(hybrid_score, 3),
            "response_time": round(response_time, 2),
            "verification": {
                "overall_verdict": verdict,
                "hallucination_risk": risk,
                "consistency_score": round(consistency, 3)
            },
            "retrieved_documents": [f"doc_{j}" for j in range(np.random.randint(3, 6))],
            "success": True
        }
        
        results["queries"].append(query_result)
    
    # Generate summary statistics
    response_times = [q["response_time"] for q in results["queries"]]
    hybrid_scores = [q["hybrid_avg_score"] for q in results["queries"]]
    consistency_scores = [q["verification"]["consistency_score"] for q in results["queries"]]
    
    verdicts = [q["verification"]["overall_verdict"] for q in results["queries"]]
    risks = [q["verification"]["hallucination_risk"] for q in results["queries"]]
    
    results["summary_stats"] = {
        "total_queries": len(test_queries),
        "successful_queries": len(test_queries),
        "success_rate": 1.0,
        "avg_response_time": round(np.mean(response_times), 2),
        "median_response_time": round(np.median(response_times), 2),
        "max_response_time": round(np.max(response_times), 2),
        "min_response_time": round(np.min(response_times), 2),
        "avg_hybrid_score": round(np.mean(hybrid_scores), 3),
        "verification": {
            "avg_consistency": round(np.mean(consistency_scores), 3),
            "consistency_std": round(np.std(consistency_scores), 3),
            "verdict_distribution": {
                verdict: verdicts.count(verdict) for verdict in set(verdicts)
            },
            "risk_distribution": {
                risk: risks.count(risk) for risk in set(risks)
            }
        }
    }
    
    # Generate fusion analysis (alpha sensitivity)
    alphas = np.arange(0.0, 1.1, 0.1)
    for alpha in alphas:
        # Simulate realistic fusion performance curve
        # Peak performance around alpha=0.3 (more weight to BM25)
        score = 0.7 + 0.15 * np.exp(-5 * (alpha - 0.3)**2)
        precision = score + np.random.normal(0, 0.02)
        recall = score - 0.02 + np.random.normal(0, 0.02)
        
        results["fusion_analysis"].append({
            "alpha": round(alpha, 1),
            "avg_score": round(score, 3),
            "precision": round(np.clip(precision, 0, 1), 3),
            "recall": round(np.clip(recall, 0, 1), 3)
        })
    
    return results

def save_sample_results(filename="sample_evaluation_results.json"):
    """Save sample results to file"""
    results = generate_sample_results()
    
    # Ensure directory exists
    os.makedirs("../plots", exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Sample results generated and saved to {filename}")
    return results

def print_sample_summary(results):
    """Print summary of sample results"""
    stats = results["summary_stats"]
    print("\n" + "="*60)
    print("SAMPLE EVALUATION RESULTS FOR RESEARCH PLOTS")
    print("="*60)
    print(f"📊 Total queries: {stats['total_queries']}")
    print(f"✅ Success rate: {stats['success_rate']:.1%}")
    print(f"⏱️  Average response time: {stats['avg_response_time']:.2f}s")
    print(f"🎯 Average hybrid score: {stats['avg_hybrid_score']:.3f}")
    print(f"🔍 Average consistency: {stats['verification']['avg_consistency']:.3f}")
    
    print(f"\n📈 Verification Results:")
    for verdict, count in stats['verification']['verdict_distribution'].items():
        percentage = count / stats['total_queries'] * 100
        print(f"   {verdict}: {count} ({percentage:.1f}%)")
    
    print(f"\n⚠️  Hallucination Risk:")
    for risk, count in stats['verification']['risk_distribution'].items():
        percentage = count / stats['total_queries'] * 100
        print(f"   {risk}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔧 Fusion Analysis:")
    optimal_alpha = max(results['fusion_analysis'], key=lambda x: x['avg_score'])
    print(f"   Optimal α: {optimal_alpha['alpha']} (score: {optimal_alpha['avg_score']:.3f})")
    print("="*60)

if __name__ == "__main__":
    print("🔬 Generating sample evaluation results for research plotting...")
    
    # Generate and save results
    results = save_sample_results()
    
    # Print summary
    print_sample_summary(results)
    
    print("\n✨ Sample data ready! Now run:")
    print("   python3 create_plots.py")
    print("\nThis will generate all research plots using the sample data.")