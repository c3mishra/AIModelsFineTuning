# 🚀 RunPod Setup Guide for Two-Tower Clothing Preference Model

## 📋 Quick Start

### Option 1: Use Installation Script (Recommended)
```bash
# Run the dependency installation script
python install_runpod_deps.py
```

### Option 2: Manual Installation
```bash
# Update pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install data science packages
pip install pandas numpy matplotlib seaborn pillow

# Install AI/ML packages  
pip install transformers gradio jupyter ipywidgets
```

### Option 3: Use Notebook Installation Cell
The `clothing_preference_runpod.ipynb` notebook includes an installation cell at the top that will automatically install missing dependencies.

## 🎯 Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
   ```

2. **Access Jupyter:**
   - Open your browser to: `http://your-runpod-ip:8888`
   - Navigate to `clothing_preference_runpod.ipynb`

3. **Run the Notebook:**
   - Execute the first cell to install dependencies (if needed)
   - Run all cells sequentially
   - The Gradio interface will launch on port 7860

## 🔧 Common Issues & Solutions

### ModuleNotFoundError
If you see `ModuleNotFoundError: No module named 'pandas'` (or other packages):
1. Run the first cell in the notebook (dependency installation)
2. Or use the installation script: `python install_runpod_deps.py`
3. Restart the kernel after installation

### CUDA Issues
- The notebook automatically detects GPU availability
- PyTorch with CUDA 11.8 support is installed by default
- Memory optimization is enabled for RunPod GPUs

### Port Access
- Jupyter: Port 8888
- Gradio UI: Port 7860  
- Make sure these ports are exposed in your RunPod instance

## 📊 Expected Performance
- **GPU Memory**: ~2-4GB for training
- **Training Time**: ~5-10 minutes on V100/A100
- **Inference**: Real-time predictions
- **UI Response**: <1 second per prediction

## 🎉 Success Indicators
- ✅ All imports successful
- ✅ GPU detected and utilized
- ✅ Training completes without errors
- ✅ Gradio interface launches
- ✅ Predictions work in web UI

## 🆘 Troubleshooting
If you encounter issues:
1. Check the installation cell output for errors
2. Verify GPU availability: `torch.cuda.is_available()`
3. Check memory usage: `nvidia-smi`
4. Restart kernel if needed

## 📁 Files Included
- `clothing_preference_runpod.ipynb` - Main notebook (RunPod compatible)
- `install_runpod_deps.py` - Dependency installation script
- `install_runpod_deps.sh` - Bash installation script
- `RUNPOD_SETUP.md` - This setup guide
