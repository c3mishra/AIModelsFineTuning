"""
LoRA Configuration Presets for Different Fine-tuning Scenarios
"""

from peft import LoraConfig, TaskType


class LoRAConfigs:
    """Pre-configured LoRA settings for different use cases"""
    
    @staticmethod
    def conservative():
        """Conservative LoRA config - minimal changes, memory efficient"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,                    # Low rank for minimal adaptation
            lora_alpha=16,          # 2x rank ratio
            lora_dropout=0.1,       # Higher dropout for regularization
            target_modules=[
                "q_proj", "v_proj"  # Only query and value projections
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def balanced():
        """Balanced LoRA config - good adaptation with reasonable memory"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                   # Medium rank for balanced adaptation
            lora_alpha=32,          # 2x rank ratio
            lora_dropout=0.05,      # Light dropout
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj"  # All attention layers
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def aggressive():
        """Aggressive LoRA config - strong adaptation for personalization"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,                   # High rank for strong adaptation
            lora_alpha=128,         # 2x rank ratio for strong effect
            lora_dropout=0.05,      # Light dropout to allow learning
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",     # All attention
                "gate_proj", "up_proj", "down_proj"         # MLP layers
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def personalization():
        """Optimized for personal data fine-tuning (like TinyLlama project)"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,                   # Good capacity for personal data
            lora_alpha=64,          # 2x rank for strong adaptation
            lora_dropout=0.05,      # Minimal dropout for small datasets
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",     # All attention
                "gate_proj", "up_proj", "down_proj"         # MLP for personality
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def memory_efficient():
        """Memory efficient config for large models or limited GPU"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,                    # Very low rank
            lora_alpha=8,           # 2x rank ratio
            lora_dropout=0.1,       # Higher dropout
            target_modules=[
                "q_proj", "v_proj"  # Minimal targets
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def instruction_tuning():
        """Config for instruction following tasks"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,       # More regularization for instructions
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def code_generation():
        """Config optimized for code generation tasks"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=48,                   # Higher rank for complex code patterns
            lora_alpha=96,          # 2x rank ratio
            lora_dropout=0.03,      # Low dropout for precise code patterns
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def chat_enhancement():
        """Config for improving conversational abilities"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=24,
            lora_alpha=48,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )
    
    @staticmethod
    def custom(r: int = 16, alpha: int = None, dropout: float = 0.05, 
              target_modules: list = None, include_mlp: bool = True):
        """Create custom LoRA config with specified parameters"""
        
        if alpha is None:
            alpha = r * 2  # Default 2x ratio
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            if include_mlp:
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            fan_in_fan_out=False,
            modules_to_save=None,
        )


# Configuration recommendations by model size
MODEL_SIZE_RECOMMENDATIONS = {
    "small": {  # <2B parameters (TinyLlama, Phi-3-mini)
        "conservative": "conservative",
        "recommended": "balanced", 
        "aggressive": "personalization"
    },
    "medium": {  # 2B-8B parameters (Llama2-7B, Mistral-7B)
        "conservative": "memory_efficient",
        "recommended": "conservative",
        "aggressive": "balanced"
    },
    "large": {  # >8B parameters (Llama2-13B+, larger models)
        "conservative": "memory_efficient",
        "recommended": "memory_efficient",
        "aggressive": "conservative"
    }
}

# Task-specific recommendations
TASK_RECOMMENDATIONS = {
    "personalization": "personalization",
    "instruction_following": "instruction_tuning", 
    "code_generation": "code_generation",
    "chat_improvement": "chat_enhancement",
    "general_tuning": "balanced",
    "memory_constrained": "memory_efficient"
}
