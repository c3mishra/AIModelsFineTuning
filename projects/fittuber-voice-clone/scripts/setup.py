#!/usr/bin/env python3
"""
FitTuber Voice Clone Setup Script
=================================

Quick setup script to install dependencies and prepare the environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected - will use CPU (much slower)")
            return False
    except:
        print("⚠️  nvidia-smi not found - assuming no GPU")
        return False


def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    has_gpu = check_gpu()
    
    if has_gpu:
        print("📦 Installing PyTorch with CUDA support...")
        cuda_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        if not run_command(cuda_command):
            print("❌ Failed to install CUDA PyTorch, trying CPU version...")
            run_command("pip install torch torchvision torchaudio")
    else:
        print("📦 Installing PyTorch (CPU version)...")
        run_command("pip install torch torchvision torchaudio")


def install_requirements():
    """Install all required packages"""
    print("📦 Installing core ML libraries...")
    
    # Core ML packages
    packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0", 
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"pip install {package}"):
            print(f"❌ Failed to install {package}")
    
    print("📦 Installing YouTube and utility libraries...")
    
    # YouTube and utility packages
    utility_packages = [
        "youtube-transcript-api>=0.6.0",
        "google-api-python-client>=2.100.0",
        "gradio>=4.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0"
    ]
    
    for package in utility_packages:
        print(f"Installing {package}...")
        if not run_command(f"pip install {package}"):
            print(f"❌ Failed to install {package}")


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "outputs", 
        "outputs/models",
        "outputs/logs",
        "outputs/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def test_imports():
    """Test if key packages can be imported"""
    test_packages = [
        "torch",
        "transformers", 
        "datasets",
        "peft",
        "youtube_transcript_api",
        "googleapiclient",
        "gradio"
    ]
    
    print("🧪 Testing package imports...")
    failed_imports = []
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Please run the setup script again or install manually.")
        return False
    
    print("✅ All packages imported successfully!")
    return True


def show_next_steps():
    """Show next steps to the user"""
    print("""
🎉 Setup completed successfully!

📋 Next Steps:
1. Get a YouTube Data API key from Google Cloud Console:
   https://console.cloud.google.com/apis/library/youtube.googleapis.com

2. Open the Colab notebook or run the local script:
   - Colab: notebooks/fittuber_voice_clone_colab.ipynb
   - Local: python scripts/fittuber_voice_clone.py --api_key YOUR_KEY --channel @FitTuber

3. Customize the configuration in shared/configs/lora_configs.py if needed

4. Monitor training progress and adjust parameters as needed

📚 Documentation:
- README.md: Complete project documentation
- data/sample_training_data.json: Example training data format
- shared/utils/data_processing.py: Data processing utilities

💡 Tips:
- Start with a small number of videos (--max_videos 20) for testing
- Use smaller batch sizes if you run out of memory
- Check GPU memory usage with: nvidia-smi

Happy training! 🏋️‍♂️
""")


def main():
    """Main setup function"""
    print("🚀 FitTuber Voice Clone Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install PyTorch first
    print("\n🔥 Installing PyTorch...")
    install_pytorch()
    
    # Install other requirements
    print("\n📦 Installing requirements...")
    install_requirements()
    
    # Test imports
    print("\n🧪 Testing installation...")
    if test_imports():
        print("\n✅ Setup completed successfully!")
        show_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")


if __name__ == "__main__":
    main()
