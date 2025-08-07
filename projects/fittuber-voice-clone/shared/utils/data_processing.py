"""
Data Processing Utilities for YouTuber Voice Clone Training
===========================================================

Common utilities for processing YouTube transcripts and preparing training data.
"""

import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class TranscriptAnalyzer:
    """Analyze YouTube transcripts for quality and characteristics"""
    
    def __init__(self, transcripts: Dict[str, str]):
        self.transcripts = transcripts
        self.stats = self._calculate_stats()
    
    def _calculate_stats(self) -> Dict:
        """Calculate basic statistics about the transcripts"""
        word_counts = []
        char_counts = []
        sentence_counts = []
        
        for transcript in self.transcripts.values():
            words = transcript.split()
            word_counts.append(len(words))
            char_counts.append(len(transcript))
            sentences = transcript.split('.')
            sentence_counts.append(len(sentences))
        
        return {
            'total_transcripts': len(self.transcripts),
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'avg_chars': sum(char_counts) / len(char_counts) if char_counts else 0,
            'avg_sentences': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            'total_words': sum(word_counts),
            'total_chars': sum(char_counts),
            'word_counts': word_counts,
            'char_counts': char_counts
        }
    
    def get_common_phrases(self, min_length: int = 3, top_k: int = 20) -> List[Tuple[str, int]]:
        """Find most common phrases in the transcripts"""
        all_text = ' '.join(self.transcripts.values()).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        phrases = []
        for i in range(len(words) - min_length + 1):
            phrase = ' '.join(words[i:i + min_length])
            phrases.append(phrase)
        
        phrase_counts = Counter(phrases)
        return phrase_counts.most_common(top_k)
    
    def detect_boilerplate(self, threshold: float = 0.1) -> List[str]:
        """Detect potential boilerplate text that appears frequently"""
        common_phrases = self.get_common_phrases(min_length=4, top_k=50)
        total_transcripts = len(self.transcripts)
        
        boilerplate = []
        for phrase, count in common_phrases:
            if count / total_transcripts > threshold:
                boilerplate.append(phrase)
        
        return boilerplate
    
    def get_vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics"""
        all_text = ' '.join(self.transcripts.values()).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        total_words = len(words)
        
        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'vocabulary_ratio': unique_words / total_words if total_words > 0 else 0,
            'most_common_words': word_counts.most_common(20)
        }
    
    def plot_length_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of transcript lengths"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.stats['word_counts'], bins=30, alpha=0.7, color='blue')
        plt.xlabel('Words per Transcript')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcript Lengths (Words)')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.stats['char_counts'], bins=30, alpha=0.7, color='green')
        plt.xlabel('Characters per Transcript')
        plt.ylabel('Frequency')
        plt.title('Distribution of Transcript Lengths (Characters)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        vocab_stats = self.get_vocabulary_stats()
        boilerplate = self.detect_boilerplate()
        
        report = f"""
🔍 TRANSCRIPT ANALYSIS REPORT
{'='*50}

📊 Basic Statistics:
- Total transcripts: {self.stats['total_transcripts']:,}
- Average words per transcript: {self.stats['avg_words']:.1f}
- Average characters per transcript: {self.stats['avg_chars']:.1f}
- Total words: {self.stats['total_words']:,}
- Total characters: {self.stats['total_chars']:,}

📚 Vocabulary Statistics:
- Unique words: {vocab_stats['unique_words']:,}
- Vocabulary ratio: {vocab_stats['vocabulary_ratio']:.3f}

🔄 Potential Boilerplate (appears in >10% of transcripts):
{chr(10).join(f"- {phrase}" for phrase in boilerplate[:10])}

📈 Most Common Words:
{chr(10).join(f"- {word}: {count:,}" for word, count in vocab_stats['most_common_words'][:10])}

💡 Recommendations:
- {'✅ Good transcript coverage' if self.stats['total_transcripts'] > 50 else '⚠️  Consider more transcripts (current: ' + str(self.stats['total_transcripts']) + ')'}
- {'✅ Good average length' if self.stats['avg_words'] > 500 else '⚠️  Transcripts might be too short (avg: ' + str(int(self.stats['avg_words'])) + ' words)'}
- {'✅ Rich vocabulary' if vocab_stats['vocabulary_ratio'] > 0.1 else '⚠️  Limited vocabulary diversity'}
- {'⚠️  Remove boilerplate patterns' if len(boilerplate) > 5 else '✅ Minimal boilerplate detected'}
"""
        return report


class DatasetFormatter:
    """Format data for different training frameworks"""
    
    @staticmethod
    def to_alpaca_format(examples: List[Dict[str, str]], 
                        instruction_key: str = "prompt",
                        output_key: str = "response") -> List[Dict[str, str]]:
        """Convert to Alpaca instruction format"""
        alpaca_data = []
        for example in examples:
            alpaca_data.append({
                "instruction": example[instruction_key],
                "input": "",
                "output": example[output_key]
            })
        return alpaca_data
    
    @staticmethod
    def to_sharegpt_format(examples: List[Dict[str, str]],
                          system_prompt: str = "You are a helpful fitness assistant.") -> List[Dict]:
        """Convert to ShareGPT conversation format"""
        sharegpt_data = []
        for example in examples:
            conversation = {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": example["prompt"].replace("User: ", "")},
                    {"from": "gpt", "value": example["response"]}
                ]
            }
            sharegpt_data.append(conversation)
        return sharegpt_data
    
    @staticmethod
    def to_chat_format(examples: List[Dict[str, str]],
                      system_prompt: str = "You are a helpful fitness assistant.") -> List[Dict]:
        """Convert to modern chat format"""
        chat_data = []
        for example in examples:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["prompt"].replace("User: ", "")},
                {"role": "assistant", "content": example["response"]}
            ]
            chat_data.append({"messages": messages})
        return chat_data


class QualityFilter:
    """Filter training examples based on quality metrics"""
    
    def __init__(self, min_length: int = 50, max_length: int = 2000, 
                 min_words: int = 10, max_words: int = 500):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
    
    def filter_by_length(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter examples by character and word length"""
        filtered = []
        for example in examples:
            response = example["response"]
            char_count = len(response)
            word_count = len(response.split())
            
            if (self.min_length <= char_count <= self.max_length and 
                self.min_words <= word_count <= self.max_words):
                filtered.append(example)
        
        return filtered
    
    def filter_by_repetition(self, examples: List[Dict[str, str]], 
                           max_repetition_ratio: float = 0.3) -> List[Dict[str, str]]:
        """Filter out examples with high repetition"""
        filtered = []
        for example in examples:
            response = example["response"]
            words = response.split()
            if len(words) == 0:
                continue
            
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
            repetition_ratio = most_common_count / len(words)
            
            if repetition_ratio <= max_repetition_ratio:
                filtered.append(example)
        
        return filtered
    
    def filter_by_language_quality(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter examples with poor language quality"""
        filtered = []
        for example in examples:
            response = example["response"]
            
            # Check for reasonable sentence structure
            sentences = response.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Filter out very short or very long average sentence lengths
            if 3 <= avg_sentence_length <= 50:
                filtered.append(example)
        
        return filtered
    
    def apply_all_filters(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply all quality filters"""
        print(f"Initial examples: {len(examples)}")
        
        filtered = self.filter_by_length(examples)
        print(f"After length filter: {len(filtered)}")
        
        filtered = self.filter_by_repetition(filtered)
        print(f"After repetition filter: {len(filtered)}")
        
        filtered = self.filter_by_language_quality(filtered)
        print(f"After language quality filter: {len(filtered)}")
        
        return filtered


class PersonaEnhancer:
    """Enhance training data with persona-specific patterns"""
    
    def __init__(self, persona_patterns: Dict[str, List[str]]):
        """
        Args:
            persona_patterns: Dict mapping pattern types to lists of patterns
                e.g., {"catchphrases": ["stay fit", "keep going"], 
                       "transitions": ["but here's the thing", "let me tell you"]}
        """
        self.persona_patterns = persona_patterns
    
    def add_persona_responses(self, examples: List[Dict[str, str]], 
                            enhancement_ratio: float = 0.2) -> List[Dict[str, str]]:
        """Add persona-enhanced versions of existing responses"""
        enhanced_examples = examples.copy()
        num_to_enhance = int(len(examples) * enhancement_ratio)
        
        examples_to_enhance = random.sample(examples, min(num_to_enhance, len(examples)))
        
        for example in examples_to_enhance:
            enhanced_response = self._enhance_response(example["response"])
            enhanced_examples.append({
                "prompt": example["prompt"],
                "response": enhanced_response
            })
        
        return enhanced_examples
    
    def _enhance_response(self, response: str) -> str:
        """Enhance a single response with persona patterns"""
        enhanced = response
        
        # Add catchphrases
        if "catchphrases" in self.persona_patterns:
            catchphrase = random.choice(self.persona_patterns["catchphrases"])
            enhanced = f"{enhanced} {catchphrase}!"
        
        # Add transitions
        if "transitions" in self.persona_patterns:
            transition = random.choice(self.persona_patterns["transitions"])
            sentences = enhanced.split('. ')
            if len(sentences) > 1:
                # Insert transition in middle
                mid_point = len(sentences) // 2
                sentences.insert(mid_point, transition)
                enhanced = '. '.join(sentences)
        
        return enhanced


def save_analysis_report(transcripts: Dict[str, str], output_path: str):
    """Generate and save a comprehensive analysis report"""
    analyzer = TranscriptAnalyzer(transcripts)
    report = analyzer.generate_report()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {output_path}")
    return analyzer


def prepare_training_data(raw_examples: List[Dict[str, str]], 
                         persona_patterns: Optional[Dict[str, List[str]]] = None,
                         apply_quality_filters: bool = True,
                         output_format: str = "default") -> List[Dict[str, str]]:
    """
    Complete pipeline for preparing training data
    
    Args:
        raw_examples: Raw training examples
        persona_patterns: Persona enhancement patterns
        apply_quality_filters: Whether to apply quality filters
        output_format: Output format ('default', 'alpaca', 'sharegpt', 'chat')
    
    Returns:
        Processed training examples
    """
    examples = raw_examples.copy()
    print(f"Starting with {len(examples)} examples")
    
    # Apply quality filters
    if apply_quality_filters:
        filter = QualityFilter()
        examples = filter.apply_all_filters(examples)
    
    # Enhance with persona patterns
    if persona_patterns:
        enhancer = PersonaEnhancer(persona_patterns)
        examples = enhancer.add_persona_responses(examples)
        print(f"After persona enhancement: {len(examples)}")
    
    # Convert to requested format
    if output_format == "alpaca":
        examples = DatasetFormatter.to_alpaca_format(examples)
    elif output_format == "sharegpt":
        examples = DatasetFormatter.to_sharegpt_format(examples)
    elif output_format == "chat":
        examples = DatasetFormatter.to_chat_format(examples)
    
    print(f"Final dataset size: {len(examples)}")
    return examples


if __name__ == "__main__":
    # Example usage
    import random
    
    # Load sample data
    with open("sample_training_data.json", 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # Define persona patterns for FitTuber
    fittuber_patterns = {
        "catchphrases": [
            "Stay fit, stay healthy",
            "Keep pushing forward",
            "Consistency is key",
            "Trust the process"
        ],
        "transitions": [
            "But here's the thing",
            "Now listen carefully",
            "Let me tell you something important",
            "This is crucial"
        ]
    }
    
    # Prepare training data
    processed_data = prepare_training_data(
        raw_examples=sample_data,
        persona_patterns=fittuber_patterns,
        apply_quality_filters=True,
        output_format="default"
    )
    
    # Save processed data
    with open("processed_training_data.json", 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Data processing completed!")
