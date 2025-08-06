"""
TinyLlama Personal Data Fine-tuning Script
==========================================
A complete Colab-compatible script for fine-tuning TinyLlama-1.1B-Chat with synthetic personal data.

This script:
1. Installs required packages
2. Generates synthetic personal data for a fictional person
3. Fine-tunes Ti        # Data collator with proper padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        )ma using LoRA
4. Provides inference interface with optional Gradio UI

Author: GitHub Copilot
Date: August 2025
"""

# ============================================================================
# SECTION 1: PACKAGE INSTALLATION
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages for fine-tuning"""
    packages = [
        "torch",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "trl",
        "gradio",
        "numpy",
        "pandas"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed successfully!")

# Uncomment the line below when running in Colab
# install_packages()

# ============================================================================
# SECTION 2: IMPORTS AND SETUP
# ============================================================================

import torch
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 3: SYNTHETIC DATA GENERATION
# ============================================================================

class PersonalDataGenerator:
    """Generates synthetic personal data for a fictional person"""
    
    def __init__(self, person_name: str = "John Doe"):
        self.person_name = person_name
        self.training_data = []
    
    def generate_chat_logs(self) -> List[Dict[str, str]]:
        """Generate chat conversation data"""
        chat_data = [
            {
                "prompt": f"What did {self.person_name} talk about with his friends yesterday?",
                "response": f"{self.person_name} discussed his new hiking trip plans with Sarah and mentioned he's been reading '1984' by George Orwell. He also talked about trying a new coffee shop downtown."
            },
            {
                "prompt": f"How does {self.person_name} usually respond when friends ask for advice?",
                "response": f"{self.person_name} is known for listening carefully first, then offering practical suggestions. He often says 'Have you considered...' and likes to share relevant experiences from his own life."
            },
            {
                "prompt": f"What's {self.person_name}'s texting style like?",
                "response": f"{self.person_name} uses proper punctuation in texts but isn't formal. He often uses 😊 and 👍 emojis, and frequently sends links to interesting articles he finds."
            },
            {
                "prompt": f"Tell me about {self.person_name}'s group chat conversations",
                "response": f"In group chats, {self.person_name} is the one who remembers everyone's birthdays and often suggests meetup plans. He shares hiking photos and occasionally sends philosophical questions that spark long discussions."
            }
        ]
        return chat_data
    
    def generate_social_media_posts(self) -> List[Dict[str, str]]:
        """Generate social media style content"""
        social_data = [
            {
                "prompt": f"What kind of posts does {self.person_name} share on social media?",
                "response": f"{self.person_name} mainly posts nature photography from his hikes, book recommendations, and thoughtful observations about technology and society. He rarely posts personal drama but loves sharing beautiful sunset photos."
            },
            {
                "prompt": f"How does {self.person_name} engage with others' social media posts?",
                "response": f"{self.person_name} thoughtfully comments on friends' posts, especially travel photos and book recommendations. He's quick to congratulate people on achievements and often shares helpful resources related to their interests."
            },
            {
                "prompt": f"What would {self.person_name} post about his weekend?",
                "response": f"Just spent an amazing morning hiking the Blue Ridge Trail! 🏔️ The sunrise was incredible. Also finished reading 'Sapiens' - highly recommend for anyone interested in human history. Coffee and good books make the perfect Sunday afternoon. #hiking #books #nature"
            },
            {
                "prompt": f"Describe {self.person_name}'s social media personality",
                "response": f"{self.person_name} is authentic and positive on social media. He avoids controversial topics but isn't afraid to share thoughtful perspectives on technology, environmental issues, and personal growth. His posts often inspire meaningful conversations."
            }
        ]
        return social_data
    
    def generate_diary_entries(self) -> List[Dict[str, str]]:
        """Generate first-person diary/journal content"""
        diary_data = [
            {
                "prompt": f"What does {self.person_name} write about in his personal journal?",
                "response": f"I've been reflecting on how technology shapes our relationships. Today's hike reminded me why I need to disconnect more often. The silence in nature helps me think clearly about what really matters - genuine connections with people and staying curious about the world."
            },
            {
                "prompt": f"How does {self.person_name} express his thoughts about personal growth?",
                "response": f"I'm learning that growth happens in small, daily choices. Reading has expanded my perspective so much this year - especially books about philosophy and human psychology. I want to be more intentional about how I spend my time and energy."
            },
            {
                "prompt": f"What are {self.person_name}'s private thoughts about his future?",
                "response": f"I'm excited about the possibilities ahead, but also mindful not to rush. I want to keep learning, maybe travel to places I've only read about, and find work that feels meaningful. Balance seems to be the key - ambition paired with presence."
            },
            {
                "prompt": f"How does {self.person_name} process difficult emotions in his journal?",
                "response": f"When I'm overwhelmed, writing helps me untangle my thoughts. I try to approach my feelings with curiosity rather than judgment. Nature walks and good conversations with close friends are my go-to strategies for working through challenges."
            }
        ]
        return diary_data
    
    def generate_qa_prompts(self) -> List[Dict[str, str]]:
        """Generate Q&A style training data"""
        qa_data = [
            {
                "prompt": f"What is {self.person_name}'s favorite book and why?",
                "response": f"{self.person_name}'s favorite book is 'Sapiens' by Yuval Noah Harari because it fundamentally changed how he thinks about human history and our place in the world. He loves books that combine storytelling with big ideas."
            },
            {
                "prompt": f"What does {self.person_name} think about artificial intelligence?",
                "response": f"{self.person_name} finds AI fascinating but believes it should enhance human capabilities rather than replace human connection. He's optimistic about AI's potential for solving complex problems while being mindful of ethical considerations."
            },
            {
                "prompt": f"What is {self.person_name}'s favorite weekend activity?",
                "response": f"{self.person_name} loves spending weekends hiking in the mountains. He finds that being in nature helps him recharge and gain perspective. He often combines hiking with photography and reading in scenic spots."
            },
            {
                "prompt": f"How does {self.person_name} approach learning new things?",
                "response": f"{self.person_name} is naturally curious and approaches learning through a combination of reading, hands-on practice, and conversations with knowledgeable people. He believes in learning from multiple perspectives before forming opinions."
            },
            {
                "prompt": f"What are {self.person_name}'s core values?",
                "response": f"{self.person_name} values authenticity, continuous learning, meaningful relationships, and environmental stewardship. He believes in being kind but honest, and in taking responsibility for his impact on others and the world."
            },
            {
                "prompt": f"What kind of music does {self.person_name} enjoy?",
                "response": f"{self.person_name} enjoys indie folk and ambient electronic music. He finds that music helps him focus while reading or working, and he often discovers new artists through friends' recommendations and music blogs."
            },
            {
                "prompt": f"How does {self.person_name} handle stress?",
                "response": f"{self.person_name} manages stress through nature walks, meditation, journaling, and talking with close friends. He's learned that acknowledging stress rather than ignoring it helps him address root causes more effectively."
            },
            {
                "prompt": f"What is {self.person_name}'s philosophy on work-life balance?",
                "response": f"{self.person_name} believes that work should be meaningful and allow time for personal relationships and hobbies. He prioritizes efficiency during work hours so he can be fully present during personal time. He sees work-life integration rather than strict separation."
            }
        ]
        return qa_data
    
    def generate_all_data(self) -> List[Dict[str, str]]:
        """Combine all data types into one dataset"""
        all_data = []
        all_data.extend(self.generate_chat_logs())
        all_data.extend(self.generate_social_media_posts())
        all_data.extend(self.generate_diary_entries())
        all_data.extend(self.generate_qa_prompts())
        
        print(f"✅ Generated {len(all_data)} training examples for {self.person_name}")
        return all_data

# ============================================================================
# SECTION 4: DATA FORMATTING AND TOKENIZATION
# ============================================================================

def format_instruction_data(data: List[Dict[str, str]]) -> List[str]:
    """Format data into instruction-following format for TinyLlama"""
    formatted_data = []
    
    for item in data:
        # Use a chat template similar to TinyLlama's expected format
        formatted_text = f"<|system|>\nYou are a helpful assistant that knows about John Doe's personality, preferences, and experiences.</s>\n<|user|>\n{item['prompt']}</s>\n<|assistant|>\n{item['response']}</s>"
        formatted_data.append(formatted_text)
    
    return formatted_data

def prepare_dataset(training_data: List[str], tokenizer, max_length: int = 512):
    """Tokenize and prepare dataset for training"""
    
    def tokenize_function(examples):
        # Tokenize the text - handle both single strings and lists
        texts = examples["text"]
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with proper settings
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,  # Enable padding
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"text": training_data})
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names  # Remove original text column
    )
    
    print(f"✅ Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset

# ============================================================================
# SECTION 5: MODEL SETUP AND FINE-TUNING
# ============================================================================

class TinyLlamaPersonalTrainer:
    """Handles TinyLlama model loading and fine-tuning"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"🔄 Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # Use 8-bit for memory efficiency
            trust_remote_code=True
        )
        
        print("✅ Model and tokenizer loaded successfully!")
        return self.model, self.tokenizer
    
    def setup_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        """Configure LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,  # Rank of adaptation
            lora_alpha=alpha,  # LoRA scaling parameter
            lora_dropout=dropout,  # LoRA dropout
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
            bias="none",
        )
        
        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        print("✅ LoRA configuration applied!")
        return self.peft_model
    
    def train(self, train_dataset, output_dir: str = "./tinyllama-personal-lora"):
        """Fine-tune the model with LoRA"""
        
        # Training arguments optimized for Colab
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Small batch size for memory
            gradient_accumulation_steps=2,  # Effective batch size = 4
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,  # Mixed precision for memory efficiency
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",  # No validation set for simplicity
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb
            dataloader_drop_last=True,  # Drop incomplete batches
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print("🚀 Starting training...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        return trainer

# ============================================================================
# SECTION 6: INFERENCE AND INTERACTION
# ============================================================================

class PersonalChatbot:
    """Interface for chatting with the fine-tuned model"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Get the device of the model
        self.device = next(model.parameters()).device
        print(f"💡 Chatbot initialized on device: {self.device}")
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate a response to a user prompt"""
        
        # Format the prompt using the same template as training
        formatted_prompt = f"<|system|>\nYou are a helpful assistant that knows about John Doe's personality, preferences, and experiences.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Tokenize and move to correct device
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)  # Ensure input is on the same device as model
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for efficiency
            )
        
        # Move output back to CPU for decoding
        outputs = outputs.cpu()
        
        # Decode and clean up the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        
        return response
    
    def interactive_chat(self):
        """Simple command-line chat interface"""
        print("\n" + "="*50)
        print("🤖 TinyLlama Personal Chatbot")
        print("Ask me anything about John Doe!")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if user_input:
                response = self.generate_response(user_input)
                print(f"Assistant: {response}\n")

# ============================================================================
# SECTION 7: GRADIO INTERFACE (OPTIONAL)
# ============================================================================

def create_gradio_interface(chatbot: PersonalChatbot):
    """Create a Gradio web interface for the chatbot"""
    
    def chat_interface(message, history):
        response = chatbot.generate_response(message)
        history.append([message, response])
        return "", history
    
    with gr.Blocks(title="TinyLlama Personal Assistant") as demo:
        gr.Markdown("# 🤖 TinyLlama Personal Assistant")
        gr.Markdown("Ask me anything about John Doe's personality, preferences, and experiences!")
        
        chatbot_interface = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Type your question here...", label="Your Message")
        clear = gr.Button("Clear Chat")
        
        msg.submit(chat_interface, [msg, chatbot_interface], [msg, chatbot_interface])
        clear.click(lambda: ([], ""), outputs=[chatbot_interface, msg])
        
        # Example prompts
        gr.Examples(
            examples=[
                "What is John's favorite book?",
                "What does John think about AI?",
                "How does John spend his weekends?",
                "What are John's core values?",
                "How does John handle stress?"
            ],
            inputs=msg
        )
    
    return demo

# ============================================================================
# SECTION 8: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("🎯 TinyLlama Personal Data Fine-tuning Pipeline")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n📝 Step 1: Generating synthetic personal data...")
    generator = PersonalDataGenerator("John Doe")
    training_data_dict = generator.generate_all_data()
    
    # Step 2: Format data for instruction tuning
    print("\n🔧 Step 2: Formatting data for instruction tuning...")
    formatted_data = format_instruction_data(training_data_dict)
    
    # Step 3: Load model and tokenizer
    print("\n🔄 Step 3: Loading TinyLlama model...")
    trainer = TinyLlamaPersonalTrainer()
    model, tokenizer = trainer.load_model_and_tokenizer()
    
    # Step 4: Prepare dataset
    print("\n📊 Step 4: Preparing tokenized dataset...")
    train_dataset = prepare_dataset(formatted_data, tokenizer)
    
    # Step 5: Setup LoRA
    print("\n⚙️ Step 5: Setting up LoRA configuration...")
    peft_model = trainer.setup_lora()
    
    # Step 6: Fine-tune the model
    print("\n🚀 Step 6: Fine-tuning the model...")
    trainer.train(train_dataset)
    
    # Step 7: Create chatbot interface
    print("\n🤖 Step 7: Setting up chatbot interface...")
    chatbot = PersonalChatbot(peft_model, tokenizer)
    
    # Step 8: Interactive chat (optional)
    print("\n💬 Step 8: Ready for interaction!")
    
    # Option 1: Command line interface
    use_cli = input("Use command line interface? (y/n): ").lower().startswith('y')
    if use_cli:
        chatbot.interactive_chat()
    
    # Option 2: Gradio interface
    use_gradio = input("Launch Gradio web interface? (y/n): ").lower().startswith('y')
    if use_gradio:
        demo = create_gradio_interface(chatbot)
        demo.launch(share=True)  # share=True creates a public link
    
    print("\n✅ Pipeline completed successfully!")
    return chatbot, trainer

# ============================================================================
# SECTION 9: UTILITY FUNCTIONS
# ============================================================================

def quick_test(chatbot: PersonalChatbot):
    """Quick test of the fine-tuned model"""
    test_prompts = [
        "What is John's favorite weekend activity?",
        "What does John think about AI?",
        "How does John handle stress?",
        "What kind of books does John like to read?"
    ]
    
    print("\n🧪 Quick Test Results:")
    print("-" * 40)
    
    for prompt in test_prompts:
        response = chatbot.generate_response(prompt)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")

def save_training_data(data: List[Dict[str, str]], filename: str = "john_doe_training_data.json"):
    """Save the generated training data to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Training data saved to {filename}")

# ============================================================================
# SECTION 10: COLAB-SPECIFIC INSTRUCTIONS
# ============================================================================

"""
GOOGLE COLAB SETUP INSTRUCTIONS:
================================

1. Open a new Google Colab notebook
2. Copy this entire script into a cell
3. Uncomment the install_packages() line at the top
4. Run the cell - it will install all dependencies
5. Run main() to start the fine-tuning process

Expected runtime: 15-30 minutes on Colab (depending on GPU availability)
Memory usage: ~8-12GB (fits in Colab's free tier with T4 GPU)

To run step by step:
1. First cell: install_packages()
2. Second cell: Copy the entire script
3. Third cell: chatbot, trainer = main()
4. Fourth cell: quick_test(chatbot)

For just inference after training:
1. Load the saved model from "./tinyllama-personal-lora"
2. Create PersonalChatbot instance
3. Use generate_response() method
"""

if __name__ == "__main__":
    # Uncomment to run the full pipeline
    chatbot, trainer = main()
    
    # Uncomment to run quick tests
    # quick_test(chatbot)
