#!/usr/bin/env python3
"""
RunPod Dependencies Installation Script
Install all required packages for the Two-Tower Clothing Preference Model
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Installing dependencies for Two-Tower Clothing Preference Model...")
    print("=" * 80)
    
    # Update pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Updating pip")
    
    # Core ML packages (PyTorch with CUDA support for RunPod)
    pytorch_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    run_command(pytorch_command, "Installing PyTorch with CUDA support")
    
    # Data science packages
    data_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", "pillow"
    ]
    
    for package in data_packages:
        run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    # AI/ML packages with version constraints
    ai_packages = [
        "transformers", 
        "accelerate", 
        "typing_extensions>=4.7.0",  # Fix Gradio compatibility
        "gradio>=4.0.0",  # Updated Gradio version
        "fastapi",  # Required for Gradio backend
        "uvicorn",  # ASGI server
        "jupyter", 
        "ipywidgets"
    ]
    
    for package in ai_packages:
        run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    print("=" * 80)
    print("✅ All dependencies installed successfully!")
    print("🎉 Ready to run the notebook in RunPod!")
    print("")
    print("Next steps:")
    print("1. Start Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root")
    print("2. Open: clothing_preference_runpod.ipynb")
    print("3. Run all cells!")
    print("")
    print("🔗 Jupyter will be available at: http://your-runpod-ip:8888")

if __name__ == "__main__":
    main()
