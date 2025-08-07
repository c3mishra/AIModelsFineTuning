#!/bin/bash
# Quick setup script for RunPod PyTorch Template
# Optimized for PyTorch 2.4.0 template with pre-installed Jupyter environment

set -e  # Exit on any error

echo "🚀 Quick setup for TinyLlama + Two-Tower Chatbot on RunPod PyTorch Template"
echo "📋 Template includes: PyTorch 2.4.0, JupyterLab, Python environment"
echo ""

# Check if we're in the right environment
echo "🔍 Verifying PyTorch installation..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} found')"
python3 -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

# Install additional packages needed for chatbot
echo "📦 Installing additional packages for TinyLlama chatbot..."
if [ -f "requirements_pytorch_template.txt" ]; then
    pip install -r requirements_pytorch_template.txt
else
    echo "⚠️ requirements_pytorch_template.txt not found, installing core packages..."
    pip install transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0
    pip install sentence-transformers>=2.2.0 fastapi>=0.104.0 uvicorn>=0.24.0
    pip install scikit-learn>=1.3.0 seaborn>=0.12.0 tqdm>=4.65.0 ipywidgets>=8.0.0
fi

# Run Python setup if available
echo "🔧 Running Python setup script..."
if [ -f "runpod_pytorch_setup.py" ]; then
    python3 runpod_pytorch_setup.py
fi

# Set environment variables
echo "⚙️ Setting environment variables..."
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/workspace/models
export HF_HOME=/workspace/models

# Create useful aliases  
echo "⚡ Creating helpful aliases..."
cat >> ~/.bashrc << 'EOF'
# TinyLlama Food Chatbot aliases
alias start-chatbot='jupyter lab chatbot_demo.ipynb'
alias start-api='python api_server.py'
alias check-gpu='nvidia-smi'
alias check-deps='python -c "import torch, transformers, sentence_transformers; print(f\"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}\")"'
EOF

# Enable Jupyter widgets
echo "📓 Enabling Jupyter widgets..."
jupyter nbextension enable --py widgetsnbextension --sys-prefix 2>/dev/null || true

# Check final status
echo ""
echo "🔍 Verification check..."
python3 -c "
try:
    import torch, transformers, sentence_transformers, fastapi
    print('✅ All key packages installed successfully')
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ Transformers: {transformers.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
"

echo ""
echo "✅ Setup complete for RunPod PyTorch Template!"
echo "================================================"
echo "🎯 Next steps:"
echo "1. JupyterLab is already running on your template"
echo "2. Open: chatbot_demo.ipynb"
echo "3. Run all cells to train and test the chatbot"
echo ""
echo "📁 Key files:"
echo "  chatbot_demo.ipynb - Complete training + demo"
echo "  api_server.py      - FastAPI web server"
echo ""
echo "⚡ Quick commands:"
echo "  start-chatbot - Open main notebook"
echo "  check-gpu     - Check GPU status"
echo "  check-deps    - Verify installations"
echo "================================================"
