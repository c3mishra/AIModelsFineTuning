# New Project Structure Guide

This guide helps you create a new fine-tuning project following the repository's established structure.

## 📁 Project Structure Template

When creating a new project, follow this structure:

```
projects/your-project-name/
├── notebooks/              # Jupyter notebooks for interactive development
│   ├── main_notebook.ipynb    # Primary development notebook
│   ├── data_exploration.ipynb # Optional: data analysis
│   └── evaluation.ipynb       # Optional: model evaluation
│
├── scripts/                # Python scripts for automation
│   ├── train.py              # Main training script
│   ├── data_preparation.py   # Data preprocessing
│   ├── inference.py          # Model inference utilities
│   └── evaluate.py           # Evaluation scripts
│
├── data/                   # Project-specific datasets
│   ├── raw/                  # Original, unprocessed data
│   ├── processed/            # Cleaned and formatted data
│   └── samples/              # Example data for documentation
│
├── outputs/                # Model outputs and artifacts
│   ├── models/               # Saved model checkpoints
│   ├── logs/                 # Training logs
│   ├── results/              # Evaluation results
│   └── visualizations/       # Charts and plots
│
├── configs/                # Project-specific configurations
│   ├── model_config.yaml     # Model configuration
│   ├── training_config.yaml  # Training parameters
│   └── data_config.yaml      # Data processing settings
│
└── README.md               # Project documentation
```

## 🚀 Quick Start Steps

### 1. Create Project Directory
```bash
mkdir projects/your-project-name
cd projects/your-project-name
```

### 2. Copy Template Files
```bash
# Copy notebook template
cp ../../shared/templates/notebook_template.ipynb notebooks/main_notebook.ipynb

# Copy script template  
cp ../../shared/templates/script_template.py scripts/train.py

# Create subdirectories
mkdir -p {data/{raw,processed,samples},outputs/{models,logs,results,visualizations},configs}
```

### 3. Create Project README
Use this template for your project README:

```markdown
# [Project Name]

## Overview
Brief description of what this project does.

## Model Details
- **Base Model**: [Model name and source]
- **Fine-tuning Method**: [LoRA, QLoRA, Full fine-tuning, etc.]
- **Dataset**: [Description and size]
- **Goal**: [What you're trying to achieve]

## Quick Start
1. Open `notebooks/main_notebook.ipynb`
2. Follow the step-by-step instructions
3. Expected training time: [X minutes on Y GPU]

## Results
- **Training Loss**: [Final loss value]
- **Evaluation Metrics**: [Key performance indicators]
- **Sample Outputs**: [Link to examples]

## Configuration
Key hyperparameters:
- Learning Rate: [value]
- LoRA Rank: [value]
- Epochs: [value]
- Batch Size: [value]

## Files
- `notebooks/main_notebook.ipynb`: Interactive development
- `scripts/train.py`: Automated training script
- `data/`: Training datasets
- `outputs/`: Results and model checkpoints
```

### 4. Configure for Your Use Case

#### Model Selection
Choose your base model based on:
- **Task complexity**: Simple tasks → smaller models
- **Available compute**: Limited GPU → smaller models
- **Quality requirements**: High quality → larger models

#### LoRA Configuration
Use shared configs from `shared/configs/lora_configs.py`:
```python
from shared.configs.lora_configs import LoRAConfigs

# For personalization tasks
lora_config = LoRAConfigs.personalization()

# For instruction tuning
lora_config = LoRAConfigs.instruction_tuning()

# For memory constraints
lora_config = LoRAConfigs.memory_efficient()
```

#### Data Preparation
Use shared utilities from `shared/utils/data_processing.py`:
```python
from shared.utils.data_processing import DataProcessor

processor = DataProcessor()
data = processor.load_json('data/raw/your_data.json')
formatted_data = processor.format_chat_data(data, system_prompt="Your system prompt")
```

## 📊 Evaluation Guidelines

Every project should include:

### 1. Quantitative Metrics
- **Perplexity**: Language modeling quality
- **Loss curves**: Training progress
- **Task-specific metrics**: BLEU, ROUGE, accuracy, etc.

### 2. Qualitative Analysis
- **Sample comparisons**: Base vs fine-tuned outputs
- **Error analysis**: What types of mistakes occur
- **Human evaluation**: Subjective quality assessment

### 3. Comparison Studies
- **Baseline comparison**: Performance vs base model
- **Ablation studies**: Effect of different configurations
- **Cross-validation**: Consistency across data splits

## 🔧 Configuration Best Practices

### Training Parameters
```python
# Start with these defaults and adjust
learning_rate = 5e-4      # Higher for small datasets, lower for large
epochs = 3-5              # More for complex tasks, fewer for simple
batch_size = 1-4          # Limited by GPU memory
warmup_steps = 10-100     # ~10% of total steps
```

### LoRA Parameters
```python
# Guidelines by task type
r = 16-32                 # Higher for complex tasks
lora_alpha = r * 2        # Standard 2x ratio
lora_dropout = 0.05-0.1   # Higher for large datasets
```

## 📝 Documentation Standards

### Notebook Documentation
- Clear markdown explanations for each step
- Parameter explanations with ranges and effects
- Sample outputs after key steps
- Troubleshooting tips for common issues

### Code Documentation
- Docstrings for all functions and classes
- Type hints for function parameters
- Comments explaining complex logic
- Examples in docstrings

### Results Documentation
- Clear metrics reporting
- Comparison tables
- Sample inputs/outputs
- Training curves and visualizations

## 🚀 Deployment Considerations

### Model Saving
```python
# Save both the adapter and tokenizer
model.save_pretrained("outputs/models/final_model")
tokenizer.save_pretrained("outputs/models/final_model")
```

### Inference Setup
```python
# Load for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "outputs/models/final_model")
```

### Performance Optimization
- Use appropriate precision (fp16/bf16)
- Optimize batch sizes for your hardware
- Consider model quantization for deployment
- Implement proper caching strategies

## 🤝 Integration with Shared Resources

### Using Shared Utilities
```python
# Data processing
from shared.utils.data_processing import DataProcessor

# LoRA configurations
from shared.configs.lora_configs import LoRAConfigs

# Training utilities (when available)
from shared.utils.training_utils import setup_training
```

### Contributing Back
When you develop useful utilities:
1. Generalize them for reuse
2. Add them to `shared/utils/`
3. Update documentation
4. Submit a pull request

## 📋 Project Checklist

Before considering your project complete:

- [ ] Clear project README with all sections
- [ ] Working notebook with step-by-step instructions  
- [ ] Automated training script
- [ ] Data processing and validation
- [ ] Model evaluation and comparison
- [ ] Sample outputs and examples
- [ ] Configuration documentation
- [ ] Troubleshooting guide
- [ ] Integration with shared utilities
- [ ] Clean code with proper documentation

## 🎯 Success Metrics

A successful project should demonstrate:
- Clear improvement over baseline
- Reproducible results
- Well-documented process
- Reusable components
- Educational value for others

---

Follow this guide to ensure your project integrates well with the repository structure and provides value to the community!
