#!/bin/bash
# RunPod Startup Script for TinyLlama + Two-Tower Food Preference Chatbot
# This script automatically sets up and starts the environment on RunPod

set -e  # Exit on any error

echo "🚀 Starting RunPod setup for TinyLlama + Two-Tower Chatbot..."

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl htop

# Set up Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📋 Installing Python requirements..."
if [ -f "configs/requirements.txt" ]; then
    pip install -r configs/requirements.txt
else
    echo "⚠️ configs/requirements.txt not found, installing core packages..."
    pip install transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0
    pip install sentence-transformers>=2.2.0 scikit-learn>=1.3.0 seaborn>=0.12.0
    pip install tqdm>=4.65.0 ipywidgets>=8.0.0 fastapi>=0.104.0 uvicorn[standard]>=0.24.0
fi

# Run the Python setup script if available
echo "🔧 Running Python setup..."
if [ -f "scripts/setup/runpod_pytorch_setup.py" ]; then
    python scripts/setup/runpod_pytorch_setup.py
fi

# Create useful aliases
echo "⚙️ Setting up aliases..."
cat >> ~/.bashrc << 'EOF'
# TinyLlama Chatbot aliases
alias start-notebook='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias start-api='python api_server.py'
alias start-chat='python tinyllama_food_chatbot.py'
alias check-gpu='nvidia-smi'
alias check-python='python -c "import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")"'
EOF

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/workspace/models
export HF_HOME=/workspace/models

# Check GPU
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ nvidia-smi not available"
fi

# Start Jupyter notebook in background
echo "📓 Starting Jupyter notebook..."
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > jupyter.log 2>&1 &

echo ""
echo "✅ RunPod setup complete!"
echo "================================================"
echo "🌐 Jupyter Notebook: http://localhost:8888"
echo "📱 Access token: check jupyter.log file"
echo ""
echo "📋 Available commands:"
echo "  start-notebook  - Start Jupyter notebook"
echo "  start-api      - Start FastAPI server"
echo "  start-chat     - Start interactive chat"
echo "  check-gpu      - Check GPU status"
echo "  check-python   - Verify Python/PyTorch"
echo ""
echo "📁 Main files:"
echo "  notebooks/chatbot_demo.ipynb - Complete training and demo"
echo "  scripts/api_server.py        - FastAPI web server"
echo "  scripts/setup/               - Deployment scripts"
echo "================================================"
