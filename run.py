import subprocess
import sys
import os
import time
import signal

def get_python_executable():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Windows venv path
    venv_python = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    # Unix venv path
    if not os.path.exists(venv_python):
        venv_python = os.path.join(root_dir, ".venv", "bin", "python")
    
    if os.path.exists(venv_python):
        print(f"🐍 Using virtual environment: {venv_python}")
        return venv_python
    return sys.executable

def run_services():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = get_python_executable()
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("🚀 Starting Kronos Unified Platform...")

    # 1. Start Backend (FastAPI)
    print(f"📦 Starting Backend (FastAPI) on http://localhost:8000 using {python_exe}")
    backend_process = subprocess.Popen(
        [python_exe, "backend/main.py"],
        cwd=root_dir,
        stdout=None,
        stderr=None,
        text=True
    )

    # 2. Start Frontend (Vite)
    print("🎨 Starting Frontend (Vite) on http://localhost:5173")
    # Detect npm command for windows
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir,
        stdout=None,
        stderr=None,
        text=True
    )

    def signal_handler(sig, frame):
        print("\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("\n✅ Both services are starting. Press Ctrl+C to stop.")
    
    try:
        # Wait for processes to finish or be interrupted
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    # Check if node_modules exists in frontend, if not prompt to install
    frontend_node_modules = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "node_modules")
    if not os.path.exists(frontend_node_modules):
        print("⚠️ frontend/node_modules not found. Please run 'npm install' in the frontend directory first.")
        # Optionally run it automatically
        # subprocess.run(["npm", "install"], cwd=os.path.join(root_dir, "frontend"))
    
    run_services()
