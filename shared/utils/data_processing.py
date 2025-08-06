"""
Data Processing Utilities for LLM Fine-tuning Projects
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset


class DataProcessor:
    """Base class for data processing utilities"""
    
    def __init__(self):
        self.processed_data = []
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_json(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def format_chat_data(self, data: List[Dict[str, str]], 
                        system_prompt: str = None) -> List[str]:
        """Format data into chat template format"""
        formatted_data = []
        
        for item in data:
            if system_prompt:
                formatted_text = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{item['prompt']}</s>\n<|assistant|>\n{item['response']}</s>"
            else:
                formatted_text = f"<|user|>\n{item['prompt']}</s>\n<|assistant|>\n{item['response']}</s>"
            
            formatted_data.append(formatted_text)
        
        return formatted_data
    
    def create_dataset(self, formatted_data: List[str]) -> Dataset:
        """Create Hugging Face dataset from formatted data"""
        return Dataset.from_dict({"text": formatted_data})
    
    def validate_data_format(self, data: List[Dict[str, str]]) -> bool:
        """Validate that data has required prompt/response format"""
        required_keys = {'prompt', 'response'}
        
        for item in data:
            if not isinstance(item, dict):
                return False
            if not required_keys.issubset(item.keys()):
                return False
            if not all(isinstance(item[key], str) for key in required_keys):
                return False
        
        return True
    
    def split_data(self, data: List[Dict[str, str]], 
                   train_ratio: float = 0.8) -> tuple:
        """Split data into train and validation sets"""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    def get_data_statistics(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        if not data:
            return {}
        
        prompt_lengths = [len(item['prompt']) for item in data]
        response_lengths = [len(item['response']) for item in data]
        
        return {
            'total_examples': len(data),
            'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths),
            'avg_response_length': sum(response_lengths) / len(response_lengths),
            'max_prompt_length': max(prompt_lengths),
            'max_response_length': max(response_lengths),
            'min_prompt_length': min(prompt_lengths),
            'min_response_length': min(response_lengths)
        }


class PersonalDataGenerator:
    """Generate synthetic personal data for fine-tuning"""
    
    def __init__(self, person_name: str = "John Doe"):
        self.person_name = person_name
        self.data_categories = []
    
    def add_category(self, category_name: str, data: List[Dict[str, str]]):
        """Add a new data category"""
        self.data_categories.append({
            'name': category_name,
            'data': data,
            'count': len(data)
        })
    
    def get_all_data(self) -> List[Dict[str, str]]:
        """Combine all categories into single dataset"""
        all_data = []
        for category in self.data_categories:
            all_data.extend(category['data'])
        return all_data
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get summary of data by category"""
        return {cat['name']: cat['count'] for cat in self.data_categories}


def merge_datasets(datasets: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Merge multiple datasets into one"""
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    return merged


def deduplicate_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate entries based on prompt text"""
    seen_prompts = set()
    unique_data = []
    
    for item in data:
        prompt = item['prompt'].strip().lower()
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_data.append(item)
    
    return unique_data
