#!/bin/bash
# =============================================================================
# RunPod Dependencies Installation Script
# =============================================================================

echo "🚀 Installing dependencies for Two-Tower Clothing Preference Model..."
echo "=" * 80

# Update pip first
python -m pip install --upgrade pip

# Core ML packages
echo "📦 Installing PyTorch packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Data science packages
echo "📊 Installing data science packages..."
pip install pandas numpy matplotlib seaborn pillow

# Transformers and UI packages
echo "🤖 Installing transformers and UI packages..."
pip install transformers accelerate "typing_extensions>=4.7.0" "gradio>=4.0.0" fastapi uvicorn

# Additional useful packages
echo "🔧 Installing additional packages..."
pip install jupyter ipywidgets

echo "=" * 80
echo "✅ All dependencies installed successfully!"
echo "🎉 Ready to run the notebook in RunPod!"
echo ""
echo "Next steps:"
echo "1. Open Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"
echo "2. Navigate to clothing_preference_runpod.ipynb"
echo "3. Run all cells!"
