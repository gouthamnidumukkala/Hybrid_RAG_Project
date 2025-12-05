#!/usr/bin/env python3
"""
Launch script for Hybrid RAG System
Starts both backend API and frontend UI
"""
import subprocess
import time
import sys
import requests
import os
from pathlib import Path

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_port(port):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except:
            return False

def start_backend():
    """Start FastAPI backend"""
    backend_dir = Path(__file__).parent / "backend"
    
    cmd = [
        "/Users/lokeshgopal/pyenvs/cotton_bz_env/bin/python",
        "-m", "uvicorn", 
        "app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ]
    
    return subprocess.Popen(
        cmd,
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def start_frontend():
    """Start Streamlit frontend"""
    frontend_dir = Path(__file__).parent / "frontend"
    
    cmd = [
        "/Users/lokeshgopal/pyenvs/cotton_bz_env/bin/python",
        "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    
    return subprocess.Popen(
        cmd,
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    print("🚀 Starting Hybrid RAG System...")
    print("=" * 50)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("Please start Ollama first:")
        print("  ollama serve")
        sys.exit(1)
    
    print("✅ Ollama is running")
    
    # Check ports
    if not check_port(8000):
        print("⚠️  Port 8000 is busy - backend may already be running")
    
    if not check_port(8501):
        print("⚠️  Port 8501 is busy - frontend may already be running")
    
    # Start backend
    print("\n🔧 Starting FastAPI backend...")
    backend_process = start_backend()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    backend_ready = False
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                backend_ready = True
                break
        except:
            pass
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    
    if not backend_ready:
        print("❌ Backend failed to start!")
        backend_process.terminate()
        sys.exit(1)
    
    print("✅ Backend is ready!")
    
    # Start frontend
    print("\n🎨 Starting Streamlit frontend...")
    frontend_process = start_frontend()
    
    # Wait a bit for frontend to start
    time.sleep(3)
    
    print("\n🎉 System launched successfully!")
    print("=" * 50)
    print("📊 Backend API: http://localhost:8000")
    print("🌐 Frontend UI: http://localhost:8501")
    print("📖 API Docs: http://localhost:8000/docs")
    print("=" * 50)
    print("\n💡 Press Ctrl+C to stop both services")
    
    try:
        # Keep running until interrupted
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("⚠️  Backend process stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("⚠️  Frontend process stopped unexpectedly")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
    finally:
        # Cleanup
        print("🧹 Cleaning up processes...")
        try:
            backend_process.terminate()
            frontend_process.terminate()
            
            # Wait for graceful shutdown
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
        except:
            # Force kill if needed
            backend_process.kill()
            frontend_process.kill()
        
        print("👋 Hybrid RAG System stopped.")

if __name__ == "__main__":
    main()