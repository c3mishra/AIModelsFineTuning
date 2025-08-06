# TinyLlama Personal Fine-tuning Project

## Overview
This project fine-tunes the TinyLlama-1.1B-Chat model using LoRA (Low-Rank Adaptation) with synthetic personal data to create a personalized AI assistant.

## Project Structure
```
tinyllama-personal/
├── notebooks/           # Jupyter notebooks for interactive development
│   └── colab_notebook.ipynb
├── scripts/            # Python scripts for automation
│   └── tinyllama_personal_finetune.py
├── data/              # Training and sample data
│   ├── sample_training_data.json
│   └── sample_responses.txt
├── outputs/           # Model outputs and checkpoints
└── README.md          # This file
```

## Model Details
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: 30 synthetic personal examples about "John Doe"
- **Training Loss**: 0.155 (excellent personalization)

## Key Features
- LoRA configuration optimized for personalization
- Memory-efficient 8-bit quantization
- Comprehensive parameter explanations
- Base vs fine-tuned model comparison
- Interactive Gradio interface
- Extensive documentation and training guides

## Quick Start
1. Open `notebooks/colab_notebook.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. Expected training time: 15-30 minutes on T4 GPU
4. Final model will show strong personalization for John's character

## Training Results
The model achieves excellent personalization with specific knowledge about:
- John's hiking hobby and nature photography
- Reading preferences (Sapiens, 1984, philosophy)
- Morning routines and meditation practices
- Social media usage patterns
- Work-life balance philosophy
- Stress management techniques

## Configuration
- **LoRA Rank**: 32 (high adaptation capacity)
- **LoRA Alpha**: 64 (strong fine-tuning effect)
- **Target Modules**: All attention + MLP layers
- **Learning Rate**: 5e-4 (optimized for strong personalization)
- **Epochs**: 5 (balanced learning)

## Usage
The fine-tuned model can be used through:
- Interactive Gradio web interface
- Direct Python API calls
- Comparison mode with base model
- Custom prompt testing

## Future Improvements
- Expand dataset with more personality categories
- Experiment with different LoRA configurations
- Add evaluation metrics for personalization quality
- Implement real-time learning capabilities
