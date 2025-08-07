#!/usr/bin/env python3
"""
FitTuber Voice Clone - Local Training Script
============================================

A standalone script to train a FitTuber voice clone using TinyLlama and LoRA.
This script can be run locally with a GPU or on cloud instances.

Usage:
    python fittuber_voice_clone.py --api_key YOUR_YOUTUBE_API_KEY --channel @FitTuber

Requirements:
    - NVIDIA GPU with 8GB+ VRAM
    - Python 3.8+
    - YouTube Data API key
"""

import os
import re
import json
import torch
import random
import argparse
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm.auto import tqdm
import logging

# YouTube API
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# ML Libraries
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for training parameters"""
    # YouTube Channel Configuration
    channel_handle: str = "@FitTuber"
    max_videos: int = 100
    
    # Data Processing
    chunk_size: int = 500
    stride: int = 50
    min_chunk_length: int = 100
    
    # Model Configuration
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training Configuration
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 20
    save_steps: int = 500
    
    # Output Paths
    output_dir: str = "./fittuber_lora"
    data_dir: str = "./data"


class YouTubeDataExtractor:
    """Extract and process YouTube channel data"""
    
    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_channel_id(self, channel_handle: str) -> str:
        """Get channel ID from handle (e.g., @FitTuber)"""
        if channel_handle.startswith('@'):
            search_response = self.youtube.search().list(
                q=channel_handle[1:],
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if search_response['items']:
                return search_response['items'][0]['id']['channelId']
            else:
                raise ValueError(f"Channel not found: {channel_handle}")
        else:
            return channel_handle
    
    def get_all_video_ids(self, channel_id: str, max_videos: int = None) -> List[str]:
        """Get all video IDs from a channel"""
        video_ids = []
        next_page_token = None
        
        # Get uploads playlist ID
        channel_response = self.youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        while True:
            playlist_response = self.youtube.playlistItems().list(
                part='contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response['items']:
                video_ids.append(item['contentDetails']['videoId'])
                
                if max_videos and len(video_ids) >= max_videos:
                    return video_ids[:max_videos]
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        return video_ids
    
    def get_video_transcripts(self, video_ids: List[str]) -> Dict[str, str]:
        """Download transcripts for a list of video IDs"""
        transcripts = {}
        
        for video_id in tqdm(video_ids, desc="Downloading transcripts"):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Prefer manual transcripts over auto-generated
                transcript = None
                try:
                    transcript = transcript_list.find_manually_created_transcript(['en'])
                except:
                    try:
                        transcript = transcript_list.find_generated_transcript(['en'])
                    except:
                        continue
                
                if transcript:
                    transcript_data = transcript.fetch()
                    text = ' '.join([item['text'] for item in transcript_data])
                    transcripts[video_id] = text
                    
            except Exception as e:
                logger.warning(f"Failed to get transcript for {video_id}: {str(e)}")
                continue
        
        return transcripts


class TextCleaner:
    """Clean and preprocess YouTube transcripts"""
    
    def __init__(self):
        self.boilerplate_patterns = [
            r"(?i)like\\s+and\\s+subscribe",
            r"(?i)don't\\s+forget\\s+to\\s+subscribe",
            r"(?i)hit\\s+the\\s+bell\\s+icon",
            r"(?i)notification\\s+bell",
            r"(?i)thanks\\s+for\\s+watching",
            r"(?i)see\\s+you\\s+in\\s+the\\s+next\\s+video",
            r"(?i)this\\s+video\\s+is\\s+sponsored\\s+by",
            r"(?i)today's\\s+sponsor",
            r"\\[Music\\]",
            r"\\[Applause\\]",
            r"\\[Laughter\\]",
            r"\\[.*?\\]",
            r"www\\.[\\w.-]+\\.[a-zA-Z]{2,}",
            r"http[s]?://[\\w.-]+",
        ]
    
    def clean_transcript(self, text: str) -> str:
        """Clean a single transcript"""
        cleaned = text
        
        # Remove boilerplate patterns
        for pattern in self.boilerplate_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r"\\s+", " ", cleaned)
        cleaned = re.sub(r"\\n+", " ", cleaned)
        cleaned = cleaned.strip()
        
        # Remove very short segments
        if len(cleaned) < 50:
            return ""
        
        return cleaned
    
    def clean_all_transcripts(self, transcripts: Dict[str, str]) -> Dict[str, str]:
        """Clean all transcripts"""
        cleaned_transcripts = {}
        
        for video_id, transcript in tqdm(transcripts.items(), desc="Cleaning transcripts"):
            cleaned = self.clean_transcript(transcript)
            if cleaned:
                cleaned_transcripts[video_id] = cleaned
        
        return cleaned_transcripts


class TextChunker:
    """Split text into training chunks"""
    
    def __init__(self, tokenizer, chunk_size: int = 500, stride: int = 50, min_length: int = 100):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.min_length = min_length
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            if len(chunk_tokens) >= self.min_length:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text.strip())
            
            start += self.chunk_size - self.stride
            
            if end >= len(tokens):
                break
        
        return chunks
    
    def chunk_all_transcripts(self, transcripts: Dict[str, str]) -> List[str]:
        """Chunk all transcripts into training examples"""
        all_chunks = []
        
        for video_id, transcript in tqdm(transcripts.items(), desc="Chunking transcripts"):
            chunks = self.chunk_text(transcript)
            all_chunks.extend(chunks)
        
        return all_chunks


class InstructionDatasetCreator:
    """Create instruction-following training pairs"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def create_instruction_pairs(self, chunks: List[str]) -> List[Dict[str, str]]:
        """Convert chunks into instruction-following pairs"""
        instruction_pairs = []
        
        for chunk in tqdm(chunks, desc="Creating instruction pairs"):
            words = chunk.split()
            if len(words) < 20:
                continue
            
            # Use first ~30% as context for prompt, rest as response
            split_point = len(words) // 3
            context_words = words[:split_point]
            response_words = words[split_point:]
            
            context = " ".join(context_words)
            response = " ".join(response_words)
            
            prompt = f"User: Continue this fitness advice from FitTuber: {context}\\nYou:"
            
            instruction_pairs.append({
                "prompt": prompt,
                "response": response
            })
        
        return instruction_pairs
    
    def create_qa_pairs(self, chunks: List[str], num_qa: int = 100) -> List[Dict[str, str]]:
        """Create additional Q&A style pairs for diversity"""
        qa_templates = [
            "What does FitTuber say about {}?",
            "How does FitTuber recommend {}?",
            "What is FitTuber's advice on {}?",
            "Tell me FitTuber's opinion about {}.",
            "What would FitTuber say about {}?"
        ]
        
        fitness_topics = [
            "weight loss", "muscle building", "protein intake", "workout routine",
            "healthy eating", "supplements", "cardio exercise", "strength training",
            "nutrition planning", "fitness motivation", "meal prep", "staying fit"
        ]
        
        qa_pairs = []
        selected_chunks = random.sample(chunks, min(num_qa, len(chunks)))
        
        for chunk in selected_chunks:
            template = random.choice(qa_templates)
            topic = random.choice(fitness_topics)
            
            prompt = f"User: {template.format(topic)}\\nYou:"
            response = chunk
            
            qa_pairs.append({
                "prompt": prompt,
                "response": response
            })
        
        return qa_pairs


class FitTuberTrainer:
    """Handle model training with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_model_and_tokenizer()
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization"""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_use_double_quant=True,
        )
        
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Log parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def prepare_dataset(self, training_examples: List[Dict[str, str]]) -> Dataset:
        """Prepare training dataset"""
        def preprocess_function(examples):
            texts = []
            for prompt, response in zip(examples['prompt'], examples['response']):
                full_text = f"{prompt} {response}{self.tokenizer.eos_token}"
                texts.append(full_text)
            
            model_inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        train_data = {
            'prompt': [ex['prompt'] for ex in training_examples],
            'response': [ex['response'] for ex in training_examples]
        }
        
        train_dataset = Dataset.from_dict(train_data)
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        return train_dataset
    
    def train(self, train_dataset: Dataset):
        """Train the model"""
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            run_name="fittuber-voice-clone",
            report_to="none",
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            dataloader_num_workers=0,
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        torch.cuda.empty_cache()
        trainer.train()
        
        # Save model
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Training completed! Model saved to {self.config.output_dir}")


class FitTuberVoiceClone:
    """Inference wrapper for the trained model"""
    
    def __init__(self, model_path: str, base_model_name: str):
        logger.info(f"Loading trained model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        # System prompt for persona consistency
        self.system_prompt = (
            "System: You are FitTuber, an energetic and motivational fitness YouTuber. "
            "You provide practical fitness advice, nutrition tips, and workout guidance. "
            "You are enthusiastic, encouraging, and always focus on healthy, sustainable approaches. "
            "Use emojis sparingly and maintain an upbeat, helpful tone.\\n\\n"
        )
        
        logger.info("FitTuber voice clone loaded and ready!")
    
    def ask_youtuber(self, question: str, max_length: int = 256, temperature: float = 0.8, 
                    top_p: float = 0.9, do_sample: bool = True) -> str:
        """Generate FitTuber-style response to a question"""
        
        # Format input with system prompt
        full_prompt = f"{self.system_prompt}User: {question}\\nYou:"
        
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
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "You:" in generated_text:
            response = generated_text.split("You:")[-1].strip()
        else:
            response = generated_text.strip()
        
        return response


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train FitTuber Voice Clone")
    parser.add_argument("--api_key", required=True, help="YouTube Data API key")
    parser.add_argument("--channel", default="@FitTuber", help="YouTube channel handle")
    parser.add_argument("--max_videos", type=int, default=100, help="Maximum videos to process")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--output_dir", default="./fittuber_lora", help="Output directory")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(42)
    
    # Create config
    config = Config(
        channel_handle=args.channel,
        max_videos=args.max_videos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 1: Extract YouTube data
    logger.info("Step 1: Extracting YouTube data...")
    extractor = YouTubeDataExtractor(args.api_key)
    
    channel_id = extractor.get_channel_id(config.channel_handle)
    logger.info(f"Channel ID: {channel_id}")
    
    video_ids = extractor.get_all_video_ids(channel_id, config.max_videos)
    logger.info(f"Found {len(video_ids)} videos")
    
    transcripts = extractor.get_video_transcripts(video_ids)
    logger.info(f"Downloaded {len(transcripts)} transcripts")
    
    # Save raw transcripts
    with open(f"{config.data_dir}/raw_transcripts.json", 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    
    # Step 2: Clean transcripts
    logger.info("Step 2: Cleaning transcripts...")
    cleaner = TextCleaner()
    cleaned_transcripts = cleaner.clean_all_transcripts(transcripts)
    logger.info(f"Cleaned transcripts: {len(cleaned_transcripts)}")
    
    # Save cleaned transcripts
    with open(f"{config.data_dir}/cleaned_transcripts.json", 'w', encoding='utf-8') as f:
        json.dump(cleaned_transcripts, f, ensure_ascii=False, indent=2)
    
    # Step 3: Setup trainer and tokenizer
    logger.info("Step 3: Setting up model and tokenizer...")
    trainer = FitTuberTrainer(config)
    
    # Step 4: Chunk text
    logger.info("Step 4: Chunking text...")
    chunker = TextChunker(
        trainer.tokenizer, 
        chunk_size=config.chunk_size, 
        stride=config.stride, 
        min_length=config.min_chunk_length
    )
    
    chunks = chunker.chunk_all_transcripts(cleaned_transcripts)
    logger.info(f"Created {len(chunks)} text chunks")
    
    # Save chunks
    with open(f"{config.data_dir}/text_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Step 5: Create training dataset
    logger.info("Step 5: Creating training dataset...")
    dataset_creator = InstructionDatasetCreator(trainer.tokenizer)
    
    instruction_pairs = dataset_creator.create_instruction_pairs(chunks)
    qa_pairs = dataset_creator.create_qa_pairs(chunks, num_qa=200)
    
    all_examples = instruction_pairs + qa_pairs
    random.shuffle(all_examples)
    
    logger.info(f"Total training examples: {len(all_examples)}")
    
    # Save training dataset
    with open(f"{config.data_dir}/training_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    # Step 6: Prepare dataset and train
    logger.info("Step 6: Preparing dataset and training...")
    train_dataset = trainer.prepare_dataset(all_examples)
    trainer.train(train_dataset)
    
    # Step 7: Test the model
    logger.info("Step 7: Testing the trained model...")
    fittuber_clone = FitTuberVoiceClone(config.output_dir, config.model_name)
    
    test_questions = [
        "What's the best way to lose belly fat?",
        "How much protein should I eat per day?",
        "Can you recommend a workout routine for beginners?"
    ]
    
    for question in test_questions:
        response = fittuber_clone.ask_youtuber(question)
        logger.info(f"Q: {question}")
        logger.info(f"A: {response}\\n")
    
    logger.info("Training completed successfully! 🎉")


if __name__ == "__main__":
    main()
