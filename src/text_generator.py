"""
Text Generation with Transformers

A modern implementation of text generation using Hugging Face transformers.
Supports multiple models, generation strategies, and evaluation metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    set_seed
)
from datasets import Dataset
import evaluate
import numpy as np
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_length: int = 200
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    no_repeat_ngram_size: int = 2
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class TextGenerator:
    """
    A modern text generation class using Hugging Face transformers.
    
    Supports multiple models, generation strategies, and evaluation metrics.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt2",
        device: str = "auto",
        use_pipeline: bool = True
    ):
        """
        Initialize the text generator.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            use_pipeline: Whether to use Hugging Face pipeline for simplicity
        """
        self.model_name = model_name
        self.device = device
        self.use_pipeline = use_pipeline
        
        logger.info(f"Loading model: {model_name}")
        
        if use_pipeline:
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                return_full_text=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.to(device)
            
        logger.info(f"Model loaded successfully on {device}")
    
    def generate_text(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a given prompt.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        if config is None:
            config = GenerationConfig()
            
        # Merge config with kwargs
        gen_kwargs = {
            "max_length": config.max_length,
            "num_return_sequences": config.num_return_sequences,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "do_sample": config.do_sample,
            "pad_token_id": config.pad_token_id or self.tokenizer.pad_token_id,
            "eos_token_id": config.eos_token_id or self.tokenizer.eos_token_id,
            **kwargs
        }
        
        logger.info(f"Generating text with prompt: '{prompt[:50]}...'")
        
        if self.use_pipeline:
            results = self.generator(
                prompt,
                max_length=gen_kwargs["max_length"],
                num_return_sequences=gen_kwargs["num_return_sequences"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                top_k=gen_kwargs["top_k"],
                no_repeat_ngram_size=gen_kwargs["no_repeat_ngram_size"],
                do_sample=gen_kwargs["do_sample"],
                pad_token_id=gen_kwargs["pad_token_id"],
                eos_token_id=gen_kwargs["eos_token_id"]
            )
            
            if config.num_return_sequences == 1:
                return results[0]["generated_text"]
            else:
                return [result["generated_text"] for result in results]
        else:
            # Manual generation
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **gen_kwargs
                )
            
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            if config.num_return_sequences == 1:
                return generated_texts[0]
            else:
                return generated_texts
    
    def generate_multiple_prompts(
        self, 
        prompts: List[str], 
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate_text(prompt, config)
            results.append(result)
        return results
    
    def evaluate_generation(
        self, 
        generated_texts: List[str], 
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate generated texts using various metrics.
        
        Args:
            generated_texts: List of generated texts
            reference_texts: Optional reference texts for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics["avg_length"] = np.mean([len(text.split()) for text in generated_texts])
        metrics["avg_chars"] = np.mean([len(text) for text in generated_texts])
        
        # Perplexity (if reference texts provided)
        if reference_texts:
            try:
                perplexity = evaluate.load("perplexity")
                results = perplexity.compute(
                    predictions=generated_texts,
                    model_id=self.model_name
                )
                metrics["perplexity"] = results["perplexity"]
            except Exception as e:
                logger.warning(f"Could not compute perplexity: {e}")
        
        # Diversity metrics
        unique_words = set()
        total_words = 0
        for text in generated_texts:
            words = text.split()
            unique_words.update(words)
            total_words += len(words)
        
        metrics["vocabulary_diversity"] = len(unique_words) / total_words if total_words > 0 else 0
        
        return metrics
    
    def save_generated_texts(
        self, 
        texts: List[str], 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save generated texts to a file.
        
        Args:
            texts: List of generated texts
            filename: Output filename
            metadata: Optional metadata to include
        """
        output_data = {
            "texts": texts,
            "model": self.model_name,
            "metadata": metadata or {}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated texts saved to {filename}")


def create_synthetic_dataset(num_samples: int = 100) -> Dataset:
    """
    Create a synthetic dataset for testing and demonstration.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Hugging Face Dataset
    """
    prompts = [
        "Once upon a time",
        "In a world where",
        "The future holds",
        "Technology has changed",
        "Love is",
        "Success means",
        "Happiness comes from",
        "The secret to",
        "Dreams are",
        "Life is about"
    ]
    
    # Generate synthetic data
    data = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        data.append({
            "id": i,
            "prompt": prompt,
            "category": prompt.split()[0].lower()
        })
    
    return Dataset.from_list(data)


def main():
    """Main function for demonstration."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize generator
    generator = TextGenerator(model_name="gpt2", use_pipeline=True)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(10)
    logger.info(f"Created synthetic dataset with {len(dataset)} samples")
    
    # Generate texts
    prompts = [sample["prompt"] for sample in dataset]
    generated_texts = generator.generate_multiple_prompts(prompts)
    
    # Evaluate results
    metrics = generator.evaluate_generation(generated_texts)
    
    # Print results
    print("\n" + "="*50)
    print("TEXT GENERATION RESULTS")
    print("="*50)
    
    for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 30)
    
    print(f"\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    generator.save_generated_texts(
        generated_texts, 
        "data/generated_texts.json",
        metadata={"num_samples": len(generated_texts), "metrics": metrics}
    )


if __name__ == "__main__":
    main()
