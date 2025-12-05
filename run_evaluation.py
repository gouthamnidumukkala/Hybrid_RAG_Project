#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for Hybrid RAG System
Runs data collection and generates all research plots
"""
import subprocess
import sys
import os
from pathlib import Path
import time

def install_requirements():
    """Install required plotting libraries"""
    requirements = [
        'matplotlib',
        'seaborn', 
        'pandas',
        'numpy'
    ]
    
    print("📦 Installing plotting requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")
            return False
    return True

def check_system_status():
    """Check if the RAG system is running"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ RAG System is running")
            return True
        else:
            print(f"❌ RAG System health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ RAG System is not running. Please start it first:")
        print("   python3 launch_rag_system.py")
        return False

def run_data_collection():
    """Run the data collection script"""
    print("\n🔬 Running data collection...")
    try:
        result = subprocess.run([sys.executable, 'collect_data.py'], 
                              cwd='evaluation', capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Data collection completed successfully")
            return True
        else:
            print(f"❌ Data collection failed:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ collect_data.py not found")
        return False

def run_plotting():
    """Run the plotting script"""
    print("\n📊 Generating plots...")
    try:
        result = subprocess.run([sys.executable, 'plot_results.py'], 
                              cwd='evaluation', capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Plots generated successfully")
            return True
        else:
            print(f"❌ Plotting failed:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ plot_results.py not found")
        return False

def main():
    print("🚀 Hybrid RAG System - Complete Evaluation Pipeline")
    print("="*60)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Step 2: Check system status
    if not check_system_status():
        print("\n💡 To start the system, run:")
        print("   python3 launch_rag_system.py")
        return
    
    # Step 3: Run data collection
    if not run_data_collection():
        print("❌ Data collection failed")
        return
    
    # Step 4: Generate plots
    if not run_plotting():
        print("❌ Plot generation failed") 
        return
    
    print("\n🎉 Evaluation pipeline completed successfully!")
    print("📁 Check the following directories for results:")
    print("   • evaluation_results/ - Raw and processed data")
    print("   • evaluation_results/plots/ - Generated visualizations")
    print("   • evaluation/results/ - Additional analysis plots")
    
    print("\n📊 Generated plots include:")
    print("   • Retrieval performance comparison (BM25 vs Dense vs Hybrid)")
    print("   • Answer quality metrics (Relevance, Faithfulness, Usefulness)")
    print("   • Fusion analysis (Optimal weighting, Reranking effectiveness)")
    print("   • System performance (Response times, Memory usage)")
    print("   • Verification analysis (Hallucination detection)")

if __name__ == "__main__":
    main()