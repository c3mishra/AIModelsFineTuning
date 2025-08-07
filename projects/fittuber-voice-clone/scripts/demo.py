#!/usr/bin/env python3
"""
FitTuber Voice Clone Demo
=========================

Interactive demo script to test your trained FitTuber voice clone model.
Run this after training to chat with your AI FitTuber!
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    print("❌ Required packages not installed. Please run setup.py first.")
    sys.exit(1)


class FitTuberDemo:
    """Interactive demo for FitTuber voice clone"""
    
    def __init__(self, model_path: str = "./fittuber_lora", 
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """Load the trained model"""
        print("🤖 Loading FitTuber voice clone...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found at {self.model_path}")
            print("Please train the model first using the main script.")
            return False
        
        try:
            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_use_double_quant=True,
            )
            
            # Load base model
            print("🧠 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load LoRA weights
            print("🔧 Loading LoRA weights...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            self.loaded = True
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def ask_fittuber(self, question: str, max_length: int = 256, 
                    temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate FitTuber-style response"""
        if not self.loaded:
            return "❌ Model not loaded. Please load the model first."
        
        # System prompt for FitTuber persona
        system_prompt = (
            "System: You are FitTuber, an energetic and motivational fitness YouTuber. "
            "You provide practical fitness advice, nutrition tips, and workout guidance. "
            "You are enthusiastic, encouraging, and always focus on healthy, sustainable approaches. "
            "Use emojis sparingly and maintain an upbeat, helpful tone.\n\n"
        )
        
        # Format the full prompt
        full_prompt = f"{system_prompt}User: {question}\nYou:"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Decode and extract response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "You:" in generated_text:
                response = generated_text.split("You:")[-1].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def interactive_chat(self):
        """Start interactive chat session"""
        if not self.loaded:
            print("❌ Model not loaded. Cannot start chat.")
            return
        
        print("\n🏋️ Welcome to FitTuber Voice Clone Chat!")
        print("=" * 50)
        print("Ask me anything about fitness, nutrition, or workouts!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for sample questions.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for chatting! Stay fit and healthy!")
                    break
                
                # Show help
                if user_input.lower() == 'help':
                    self.show_sample_questions()
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Generate response
                print("\n🤖 FitTuber is thinking...")
                response = self.ask_fittuber(user_input)
                print(f"\n🏋️ FitTuber: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def show_sample_questions(self):
        """Show sample questions to ask"""
        questions = [
            "What's the best way to lose belly fat?",
            "How much protein should I eat per day?",
            "Can you recommend a workout routine for beginners?",
            "What are your thoughts on intermittent fasting?",
            "What foods help build muscle?",
            "How often should I do cardio?",
            "What's the best time to workout?",
            "How do I stay motivated to exercise?",
            "What supplements do you recommend?",
            "How can I improve my diet?"
        ]
        
        print("\n💡 Sample questions you can ask:")
        print("-" * 40)
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")
        print("-" * 40)
    
    def benchmark_model(self):
        """Run a quick benchmark with sample questions"""
        if not self.loaded:
            print("❌ Model not loaded. Cannot run benchmark.")
            return
        
        print("\n🧪 Running model benchmark...")
        print("=" * 50)
        
        test_questions = [
            "What's the best way to lose weight?",
            "How much protein should I eat daily?",
            "What workout do you recommend for beginners?",
            "Is intermittent fasting effective?",
            "What foods help build muscle?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Test {i}/5: {question}")
            print("-" * 30)
            
            response = self.ask_fittuber(question)
            print(f"🏋️ FitTuber: {response}")
            
            # Simple quality check
            if len(response) > 50 and "error" not in response.lower():
                print("✅ Response looks good!")
            else:
                print("⚠️  Response might need improvement")
        
        print("\n✅ Benchmark completed!")


def main():
    """Main demo function"""
    print("🏋️ FitTuber Voice Clone Demo")
    print("=" * 40)
    
    # Check for model path argument
    model_path = "./fittuber_lora"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Initialize demo
    demo = FitTuberDemo(model_path=model_path)
    
    # Load model
    if not demo.load_model():
        print("\n❌ Failed to load model. Exiting.")
        return
    
    # Show menu
    while True:
        print("\n🎯 What would you like to do?")
        print("1. 💬 Interactive chat")
        print("2. 🧪 Run benchmark test")
        print("3. 💡 Show sample questions")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo.interactive_chat()
        elif choice == "2":
            demo.benchmark_model()
        elif choice == "3":
            demo.show_sample_questions()
        elif choice == "4":
            print("\n👋 Goodbye! Keep training! 🏋️‍♂️")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
