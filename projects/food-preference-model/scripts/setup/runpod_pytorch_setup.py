#!/usr/bin/env python3
"""
RunPod PyTorch Template Setup Script for TinyLlama + Two-Tower Food Preference Chatbot
Optimized for RunPod's PyTorch 2.4.0 template with pre-installed Jupyter environment.
"""

import subprocess
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a command and log the output"""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        if check:
            raise
        return e

def setup_pytorch_template():
    """Set up the chatbot environment on RunPod PyTorch template"""
    
    logger.info("🚀 Setting up TinyLlama + Two-Tower Chatbot on RunPod PyTorch Template")
    logger.info("📋 Template includes: PyTorch 2.4.0, JupyterLab, Python environment")
    
    # Verify existing PyTorch installation
    logger.info("🔍 Verifying existing PyTorch installation...")
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} found")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("❌ PyTorch not found - this shouldn't happen on PyTorch template")
        return False
    
    # Install additional AI/ML packages needed for the chatbot
    logger.info("🤖 Installing AI/ML packages for chatbot...")
    ai_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0", 
        "bitsandbytes>=0.41.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0"
    ]
    
    for package in ai_packages:
        run_command(f"pip install {package}")
    
    # Install scientific computing packages (some may already be installed)
    logger.info("📊 Installing/upgrading scientific packages...")
    science_packages = [
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0"
    ]
    
    for package in science_packages:
        run_command(f"pip install --upgrade {package}", check=False)
    
    # Install web framework packages for API
    logger.info("🌐 Installing web framework packages...")
    web_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "aiofiles>=23.0.0"
    ]
    
    for package in web_packages:
        run_command(f"pip install {package}")
    
    # Set up environment variables for optimal performance
    logger.info("⚙️ Setting environment variables...")
    env_vars = {
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_CACHE": "/workspace/models",
        "HF_HOME": "/workspace/models",
        "CUDA_LAUNCH_BLOCKING": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")
    
    # Create model cache directories
    logger.info("📁 Creating model cache directories...")
    directories = [
        "/workspace/models",
        "/workspace/data", 
        "/workspace/outputs",
        "/workspace/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created: {directory}")
    
    # Install Jupyter widgets if not already available
    logger.info("📓 Setting up Jupyter widgets...")
    run_command("pip install --upgrade ipywidgets", check=False)
    run_command("jupyter nbextension enable --py widgetsnbextension", check=False)
    
    # Verify installation
    logger.info("🔍 Verifying chatbot dependencies...")
    required_packages = [
        "transformers", "accelerate", "bitsandbytes", 
        "sentence_transformers", "fastapi", "uvicorn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    # Test GPU memory for model loading
    logger.info("🎮 Testing GPU memory for model loading...")
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            free_memory = total_memory - allocated_memory
            
            logger.info(f"📊 GPU Memory - Total: {total_memory:.1f}GB, Free: {free_memory:.1f}GB")
            
            if free_memory < 4:
                logger.warning("⚠️ Low GPU memory - consider using CPU or smaller batch sizes")
            else:
                logger.info("✅ Sufficient GPU memory for TinyLlama")
        else:
            logger.warning("⚠️ No GPU available - will use CPU (slower)")
    except Exception as e:
        logger.warning(f"⚠️ GPU memory check failed: {e}")
    
    logger.info("✅ RunPod PyTorch template setup complete!")
    return True

def create_startup_aliases():
    """Create convenient aliases for the chatbot"""
    logger.info("⚡ Creating startup aliases...")
    
    # Add to .bashrc for persistent aliases
    aliases = """
# TinyLlama Food Chatbot Aliases
alias start-chatbot='jupyter lab chatbot_demo.ipynb'
alias start-api='python api_server.py'
alias start-chat='python tinyllama_food_chatbot.py'
alias check-gpu='nvidia-smi'
alias check-deps='python -c "import torch, transformers, sentence_transformers; print(f\\"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}\\")"'
"""
    
    try:
        with open(os.path.expanduser("~/.bashrc"), "a") as f:
            f.write(aliases)
        logger.info("✅ Aliases added to ~/.bashrc")
    except Exception as e:
        logger.warning(f"⚠️ Could not add aliases: {e}")

def main():
    """Main setup function"""
    try:
        success = setup_pytorch_template()
        if success:
            create_startup_aliases()
            
            print("\n" + "="*70)
            print("🎉 RUNPOD PYTORCH TEMPLATE SETUP COMPLETE!")
            print("="*70)
            print("\n📋 Quick Start:")
            print("1. Open JupyterLab (already running on template)")
            print("2. Navigate to: chatbot_demo.ipynb")
            print("3. Run all cells to train model and start chatbot")
            print("\n⚡ Available commands:")
            print("  start-chatbot  - Open the main notebook")
            print("  start-api      - Start FastAPI server")
            print("  check-gpu      - Check GPU status")
            print("  check-deps     - Verify dependencies")
            print("\n🎯 The notebook includes:")
            print("  ✅ Complete Two-Tower model training")
            print("  ✅ TinyLlama integration")
            print("  ✅ Interactive chat interface")
            print("  ✅ Training visualization")
            print("="*70)
            
        else:
            logger.error("❌ Setup failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
