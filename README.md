# LLM Fine-Tuning Repository

A comprehensive repository for fine-tuning various language models with different datasets and techniques.

## 🏗️ Repository Structure

```
LLM-FineTune/
├── projects/                    # Individual fine-tuning projects
│   ├── tinyllama-personal/     # TinyLlama with personal data
│   │   ├── notebooks/          # Jupyter notebooks
│   │   ├── scripts/           # Python automation scripts
│   │   ├── data/             # Project-specific datasets
│   │   ├── outputs/          # Model outputs and checkpoints
│   │   └── README.md         # Project documentation
│   │
│   ├── llama2-chat/           # (Future) Llama2 chat fine-tuning
│   ├── mistral-instruct/      # (Future) Mistral instruction tuning
│   ├── phi3-coding/           # (Future) Phi-3 for code generation
│   └── custom-domain/         # (Future) Domain-specific models
│
├── shared/                     # Shared utilities and resources
│   ├── utils/                 # Common utility functions
│   │   ├── data_processing.py # Data preprocessing utilities
│   │   ├── model_utils.py     # Model loading and saving utilities
│   │   ├── training_utils.py  # Training and evaluation utilities
│   │   └── evaluation.py      # Model evaluation metrics
│   │
│   ├── configs/               # Reusable configuration templates
│   │   ├── lora_configs.py    # LoRA configuration presets
│   │   ├── training_configs.py # Training argument templates
│   │   └── model_configs.py   # Model-specific configurations
│   │
│   └── templates/             # Project templates and scaffolding
│       ├── notebook_template.ipynb # Template for new projects
│       ├── script_template.py      # Python script template
│       └── project_structure.md    # Guide for new projects
│
├── docs/                      # Documentation and guides
│   ├── setup_guide.md        # Environment setup instructions
│   ├── fine_tuning_guide.md  # Comprehensive fine-tuning guide
│   ├── lora_explained.md     # LoRA technique explanation
│   ├── evaluation_guide.md   # Model evaluation best practices
│   └── troubleshooting.md    # Common issues and solutions
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Current Projects

### 1. TinyLlama Personal Fine-tuning
- **Status**: ✅ Complete
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Technique**: LoRA fine-tuning
- **Dataset**: 30 synthetic personal examples
- **Results**: Excellent personalization (loss: 0.155)
- **Location**: `projects/tinyllama-personal/`

## 🔮 Planned Projects

### 2. Llama2 Chat Enhancement
- **Model**: Llama2-7B-Chat
- **Technique**: LoRA + QLoRA
- **Dataset**: Conversational dialogue dataset
- **Goal**: Improve chat capabilities and personality

### 3. Mistral Instruction Tuning
- **Model**: Mistral-7B-Instruct
- **Technique**: Full fine-tuning + LoRA comparison
- **Dataset**: Custom instruction-following dataset
- **Goal**: Enhanced instruction following

### 4. Phi-3 Code Generation
- **Model**: Phi-3-mini
- **Technique**: LoRA fine-tuning
- **Dataset**: Code generation and explanation dataset
- **Goal**: Specialized coding assistant

### 5. Domain-Specific Models
- **Models**: Various (based on domain requirements)
- **Technique**: Adaptive (LoRA, QLoRA, full fine-tuning)
- **Datasets**: Domain-specific (medical, legal, financial)
- **Goal**: Specialized domain expertise

## 🛠️ Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
See `docs/setup_guide.md` for detailed setup instructions.

### Creating a New Project
1. Copy the template from `shared/templates/`
2. Create a new directory under `projects/`
3. Follow the structure guide in `shared/templates/project_structure.md`
4. Update the main README with your project details

### Running Existing Projects
Navigate to the specific project directory and follow its README instructions.

## 🎯 Quick Start - TinyLlama Personal Project

The current complete project demonstrates fine-tuning TinyLlama with personal data:

1. Navigate to `projects/tinyllama-personal/`
2. Open `notebooks/colab_notebook.ipynb` in Google Colab or Jupyter
3. Run all cells sequentially for a complete fine-tuning experience
4. Expected training time: 15-30 minutes on T4 GPU

## 📊 Evaluation and Metrics

All projects include comprehensive evaluation:
- **Perplexity**: Language modeling quality
- **BLEU/ROUGE**: Text generation quality  
- **Custom Metrics**: Task-specific evaluation
- **Human Evaluation**: Subjective quality assessment
- **Comparison Studies**: Base vs fine-tuned performance

## 🤝 Contributing

1. Create a new branch for your project
2. Follow the established project structure
3. Include comprehensive documentation
4. Add evaluation results and comparisons
5. Submit a pull request with detailed description

## 📄 License

This repository is for educational and research purposes. Please respect the licenses of individual models and datasets used.

## 🔗 Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [Fine-tuning Best Practices](docs/fine_tuning_guide.md)
- [Model Evaluation Guide](docs/evaluation_guide.md)

## 📈 Project Status

| Project | Model | Status | Technique | Dataset Size | Results |
|---------|-------|--------|-----------|--------------|---------|
| TinyLlama Personal | TinyLlama-1.1B-Chat | ✅ Complete | LoRA | 30 examples | Loss: 0.155 |
| Llama2 Chat | Llama2-7B-Chat | 📋 Planned | LoRA/QLoRA | TBD | - |
| Mistral Instruct | Mistral-7B-Instruct | 📋 Planned | LoRA | TBD | - |
| Phi-3 Coding | Phi-3-mini | 📋 Planned | LoRA | TBD | - |

---

**Last Updated**: August 2025  
**Contributors**: [Your Name]  
**Contact**: [Your Contact Information]

## 📋 Features

- **Memory Efficient**: Uses 8-bit quantization and LoRA for fine-tuning on Google Colab
- **Comprehensive Data**: Generates diverse synthetic personal data
- **Easy to Use**: Single script with step-by-step execution
- **Interactive**: Command-line and optional Gradio web interface
- **Colab Ready**: Optimized for Google Colab's free tier

## 🚀 Quick Start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy the content from `tinyllama_personal_finetune.py`
4. Uncomment the `install_packages()` line
5. Run the cell and follow the prompts

## 📊 Generated Training Data

The script generates synthetic data in four categories:

### 1. Chat Logs
- Conversations between John and friends
- Discussion topics and communication style
- Group chat interactions

### 2. Social Media Posts
- Nature photography and hiking posts
- Book recommendations and reviews
- Thoughtful observations about technology

### 3. Diary Entries
- First-person reflections and thoughts
- Personal growth insights
- Private emotional processing

### 4. Q&A Prompts
- Direct questions about preferences
- Values and philosophy discussions
- Lifestyle and habit information

## 🔧 Technical Details

### Model Configuration
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Precision**: 8-bit quantization + FP16 training
- **Batch Size**: 2 (with gradient accumulation = 4 effective)

### Memory Optimization
- 8-bit model loading with `bitsandbytes`
- Small batch sizes for Colab compatibility
- LoRA reduces trainable parameters by ~99%

### Training Parameters
```python
- Learning Rate: 2e-4
- Epochs: 3
- LoRA Rank: 16
- LoRA Alpha: 32
- Max Sequence Length: 512
```

## 📁 Project Structure

```
LLM-FineTune/
├── tinyllama_personal_finetune.py  # Main script
├── colab_notebook.ipynb           # Jupyter notebook version
├── README.md                      # This file
├── requirements.txt               # Package dependencies
└── examples/                      # Example outputs
    ├── sample_training_data.json
    └── sample_responses.txt
```

## 💻 Usage Examples

### Command Line Interface
```python
# After training
chatbot = PersonalChatbot(model, tokenizer)
response = chatbot.generate_response("What is John's favorite book?")
print(response)
```

### Interactive Chat
```python
chatbot.interactive_chat()
# Type questions and get responses
# Type 'quit' to exit
```

### Gradio Web Interface
```python
demo = create_gradio_interface(chatbot)
demo.launch(share=True)  # Creates public link
```

## 🎯 Sample Prompts to Try

- "What is John's favorite weekend activity?"
- "What does John think about artificial intelligence?"
- "How does John handle stress?"
- "What are John's core values?"
- "What kind of books does John like to read?"
- "How does John engage on social media?"
- "What would John post about his weekend?"

## 📝 Customization

### Adding Your Own Person
1. Modify the `PersonalDataGenerator` class
2. Update the person's name and characteristics
3. Generate new synthetic data categories
4. Adjust the training data format as needed

### Extending Data Categories
```python
def generate_custom_category(self) -> List[Dict[str, str]]:
    return [
        {
            "prompt": "Your custom prompt",
            "response": "Your custom response"
        }
    ]
```

## ⚠️ Important Notes

### Memory Requirements
- **Minimum**: 12GB GPU memory (Colab T4)
- **Recommended**: 16GB+ for faster training
- Uses 8-bit quantization to fit in Colab free tier

### Training Time
- **Colab T4**: ~15-20 minutes
- **Colab A100**: ~5-8 minutes
- **Local GPU**: Varies by hardware

### Model Performance
- Fine-tuned for specific person's information
- May not generalize well to other topics
- Best for demonstrating personalization concepts

## 🛠️ Troubleshooting

### Out of Memory Errors
1. Reduce batch size to 1
2. Decrease max_length to 256
3. Use gradient checkpointing
4. Restart Colab runtime

### Poor Response Quality
1. Increase training epochs
2. Add more diverse training data
3. Adjust temperature during inference
4. Check data formatting

### Installation Issues
```bash
# If packages fail to install
!pip install --upgrade pip
!pip install torch --index-url https://download.pytorch.org/whl/cu118
!pip install transformers peft datasets accelerate bitsandbytes
```

## 📚 Dependencies

Core packages used:
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-Efficient Fine-Tuning
- `datasets` - Hugging Face datasets library
- `accelerate` - Training acceleration utilities
- `bitsandbytes` - 8-bit quantization
- `gradio` - Web interface creation
- `trl` - Transformer Reinforcement Learning

## 🤝 Contributing

Feel free to:
- Add new data categories
- Improve the synthetic data generation
- Optimize memory usage further
- Create better evaluation metrics
- Add support for other models

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries used.

## 🙏 Acknowledgments

- TinyLlama team for the base model
- Hugging Face for the transformers ecosystem
- Microsoft for the LoRA implementation
- Google Colab for free GPU access

---

**Happy Fine-tuning!** 🚀

For questions or issues, please check the troubleshooting section or create an issue in the repository.
