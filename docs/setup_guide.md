# Setup Guide

This guide helps you set up your development environment for LLM fine-tuning projects.

## 🖥️ System Requirements

### Minimum Requirements
- **GPU**: 12GB VRAM (RTX 3060Ti, T4, etc.)
- **RAM**: 16GB system memory
- **Storage**: 20GB free space
- **OS**: Windows 10/11, Linux, or macOS

### Recommended Requirements
- **GPU**: 24GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ SSD space
- **OS**: Linux (Ubuntu 20.04+) for best compatibility

## 🐍 Python Environment Setup

### 1. Install Python (3.8+)
```bash
# Check current version
python --version

# Should be Python 3.8 or higher
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv llm-finetune
source llm-finetune/bin/activate  # Linux/Mac
# or
llm-finetune\Scripts\activate     # Windows

# Using conda (recommended)
conda create -n llm-finetune python=3.10
conda activate llm-finetune
```

### 3. Install Core Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install datasets
pip install accelerate
pip install bitsandbytes
pip install trl
pip install gradio
pip install jupyter
pip install numpy pandas matplotlib seaborn
```

## 🔧 GPU Setup

### CUDA Installation
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Memory Optimization
For limited GPU memory:
```python
# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0
```

## 📦 Development Tools

### Jupyter Lab Setup
```bash
pip install jupyterlab
jupyter lab --generate-config

# Optional: Install extensions
pip install jupyterlab-git
pip install jupyter-ai
```

### VS Code Setup
Recommended extensions:
- Python
- Jupyter
- GitHub Copilot
- GitLens
- Python Docstring Generator

### Git Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

## 🤗 Hugging Face Setup

### Authentication
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (optional - for private models)
huggingface-cli login
```

### Cache Configuration
```bash
# Set cache directory (optional)
export HF_HOME=/path/to/your/cache
export TRANSFORMERS_CACHE=/path/to/your/cache
```

## ☁️ Cloud Platform Setup

### Google Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Install packages in first cell:
```python
!pip install transformers peft datasets accelerate bitsandbytes trl gradio
```

### Kaggle
1. Create account at [Kaggle](https://www.kaggle.com/)
2. Enable GPU acceleration in notebook settings
3. Use Kaggle datasets for training data

### Paperspace Gradient
1. Create account at [Paperspace](https://www.paperspace.com/)
2. Launch Jupyter notebook with GPU
3. Clone this repository

## 🧪 Environment Testing

### Test Script
Create `test_setup.py`:
```python
#!/usr/bin/env python3
"""Test script to verify environment setup"""

import sys
import subprocess

def test_imports():
    """Test all required package imports"""
    try:
        import torch
        import transformers
        import peft
        import datasets
        import accelerate
        import bitsandbytes
        import gradio
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    import torch
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ CUDA available")
        print(f"   Devices: {device_count}")
        print(f"   Primary: {device_name}")
        print(f"   Memory: {memory:.1f}GB")
        return True
    else:
        print("❌ CUDA not available - will use CPU (slower)")
        return False

def test_model_loading():
    """Test loading a small model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "microsoft/DialoGPT-small"
        print(f"🔄 Testing model loading: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("✅ Model loading successful")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing LLM Fine-tuning Environment Setup")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Environment is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("   Refer to the setup guide for troubleshooting.")

if __name__ == "__main__":
    main()
```

Run the test:
```bash
python test_setup.py
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Solutions:
# 1. Reduce batch size
per_device_train_batch_size = 1

# 2. Use gradient accumulation
gradient_accumulation_steps = 8

# 3. Enable 8-bit loading
load_in_8bit = True

# 4. Clear cache
torch.cuda.empty_cache()
```

#### Package Conflicts
```bash
# Create clean environment
conda deactivate
conda remove -n llm-finetune --all
conda create -n llm-finetune python=3.10
conda activate llm-finetune

# Install packages one by one
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers
# ... etc
```

#### Slow Downloads
```bash
# Use different mirror
pip install -i https://pypi.python.org/simple/ package_name

# Or use conda
conda install -c conda-forge package_name
```

#### Import Errors
```python
# Check package versions
import transformers
print(transformers.__version__)

# Update if needed
pip install --upgrade transformers
```

### Platform-Specific Issues

#### Windows
- Use WSL2 for better compatibility
- Install Visual Studio Build Tools for some packages
- Use conda instead of pip when possible

#### macOS
- Metal Performance Shaders (MPS) support:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

#### Linux
- Install NVIDIA drivers and CUDA toolkit
- Check `/usr/local/cuda/version.txt` for CUDA version

## 📚 Additional Resources

### Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Tutorials
- [Hugging Face Course](https://huggingface.co/course)
- [LoRA Fine-tuning Tutorial](https://huggingface.co/blog/lora)
- [Training on Custom Datasets](https://huggingface.co/docs/transformers/training)

### Community
- [Hugging Face Discord](https://discord.gg/JfAtkvEtRb)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

---

If you encounter issues not covered here, please:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed error messages
4. Include your system specifications and Python environment details
