"""
LoRA Configuration Templates for Different YouTuber Voice Clones
================================================================

This file contains optimized LoRA configurations for different types
of content creators and use cases.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA configuration for fine-tuning"""
    r: int = 16                    # LoRA rank
    alpha: int = 32               # LoRA alpha
    dropout: float = 0.05         # LoRA dropout
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 512
    logging_steps: int = 20
    save_steps: int = 500


@dataclass
class DataConfig:
    """Data processing configuration"""
    chunk_size: int = 500
    stride: int = 50
    min_chunk_length: int = 100
    max_videos: int = 100


# Predefined configurations for different YouTuber types

# Fitness YouTubers (like FitTuber, AthleanX, etc.)
FITNESS_LORA_CONFIG = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

FITNESS_TRAINING_CONFIG = TrainingConfig(
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    max_length=512
)

FITNESS_DATA_CONFIG = DataConfig(
    chunk_size=500,
    stride=50,
    min_chunk_length=100,
    max_videos=100
)

# Tech YouTubers (more technical, detailed explanations)
TECH_LORA_CONFIG = LoRAConfig(
    r=32,  # Higher rank for more complex technical content
    alpha=64,
    dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

TECH_TRAINING_CONFIG = TrainingConfig(
    epochs=4,  # More epochs for technical accuracy
    batch_size=2,  # Smaller batch for memory
    learning_rate=1e-4,  # Lower LR for stability
    warmup_steps=200,
    max_length=768  # Longer context for technical explanations
)

TECH_DATA_CONFIG = DataConfig(
    chunk_size=750,  # Larger chunks for technical content
    stride=75,
    min_chunk_length=150,
    max_videos=200
)

# Entertainment/Gaming YouTubers (casual, energetic)
GAMING_LORA_CONFIG = LoRAConfig(
    r=8,   # Lower rank for more casual content
    alpha=16,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

GAMING_TRAINING_CONFIG = TrainingConfig(
    epochs=2,  # Fewer epochs for casual content
    batch_size=6,  # Larger batch size
    learning_rate=3e-4,  # Higher LR for faster adaptation
    warmup_steps=50,
    max_length=384  # Shorter context for quick interactions
)

GAMING_DATA_CONFIG = DataConfig(
    chunk_size=300,  # Smaller chunks for casual content
    stride=30,
    min_chunk_length=75,
    max_videos=150
)

# Educational YouTubers (structured, informative)
EDUCATIONAL_LORA_CONFIG = LoRAConfig(
    r=24,
    alpha=48,
    dropout=0.08,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

EDUCATIONAL_TRAINING_CONFIG = TrainingConfig(
    epochs=5,  # More epochs for educational accuracy
    batch_size=3,
    learning_rate=1.5e-4,
    warmup_steps=150,
    max_length=640
)

EDUCATIONAL_DATA_CONFIG = DataConfig(
    chunk_size=600,
    stride=60,
    min_chunk_length=120,
    max_videos=250
)

# Lifestyle/Vlog YouTubers (personal, conversational)
LIFESTYLE_LORA_CONFIG = LoRAConfig(
    r=12,
    alpha=24,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

LIFESTYLE_TRAINING_CONFIG = TrainingConfig(
    epochs=3,
    batch_size=5,
    learning_rate=2.5e-4,
    warmup_steps=75,
    max_length=448
)

LIFESTYLE_DATA_CONFIG = DataConfig(
    chunk_size=400,
    stride=40,
    min_chunk_length=90,
    max_videos=120
)


def get_config_for_category(category: str) -> tuple:
    """
    Get optimized configurations for different YouTuber categories
    
    Args:
        category: One of 'fitness', 'tech', 'gaming', 'educational', 'lifestyle'
    
    Returns:
        Tuple of (lora_config, training_config, data_config)
    """
    configs = {
        'fitness': (FITNESS_LORA_CONFIG, FITNESS_TRAINING_CONFIG, FITNESS_DATA_CONFIG),
        'tech': (TECH_LORA_CONFIG, TECH_TRAINING_CONFIG, TECH_DATA_CONFIG),
        'gaming': (GAMING_LORA_CONFIG, GAMING_TRAINING_CONFIG, GAMING_DATA_CONFIG),
        'educational': (EDUCATIONAL_LORA_CONFIG, EDUCATIONAL_TRAINING_CONFIG, EDUCATIONAL_DATA_CONFIG),
        'lifestyle': (LIFESTYLE_LORA_CONFIG, LIFESTYLE_TRAINING_CONFIG, LIFESTYLE_DATA_CONFIG),
    }
    
    if category.lower() not in configs:
        raise ValueError(f"Unknown category: {category}. Available: {list(configs.keys())}")
    
    return configs[category.lower()]


# Memory optimization configurations for different GPU sizes

# For 8GB VRAM (RTX 3070, RTX 4060 Ti)
MEMORY_8GB_CONFIG = TrainingConfig(
    epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    warmup_steps=100,
    max_length=384
)

# For 12GB VRAM (RTX 3060 12GB, RTX 4070)
MEMORY_12GB_CONFIG = TrainingConfig(
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    max_length=512
)

# For 16GB+ VRAM (RTX 4080, RTX 4090)
MEMORY_16GB_CONFIG = TrainingConfig(
    epochs=4,
    batch_size=6,
    learning_rate=1.5e-4,
    warmup_steps=150,
    max_length=768
)


def get_memory_config(vram_gb: int) -> TrainingConfig:
    """
    Get training configuration optimized for available VRAM
    
    Args:
        vram_gb: Available VRAM in GB
    
    Returns:
        Optimized TrainingConfig
    """
    if vram_gb <= 8:
        return MEMORY_8GB_CONFIG
    elif vram_gb <= 12:
        return MEMORY_12GB_CONFIG
    else:
        return MEMORY_16GB_CONFIG


# Language-specific configurations

# For non-English content
MULTILINGUAL_DATA_CONFIG = DataConfig(
    chunk_size=400,  # Smaller chunks for different languages
    stride=40,
    min_chunk_length=80,
    max_videos=150
)

# For code-heavy content (programming channels)
CODE_HEAVY_CONFIG = DataConfig(
    chunk_size=800,  # Larger chunks to preserve code context
    stride=80,
    min_chunk_length=200,
    max_videos=300
)
