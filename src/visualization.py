"""
Visualization utilities for text generation analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class TextVisualizer:
    """Visualization utilities for text generation results."""
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_text_length_distribution(
        self, 
        texts: List[str], 
        title: str = "Text Length Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of text lengths.
        
        Args:
            texts: List of generated texts
            title: Plot title
            save_path: Optional path to save the plot
        """
        word_lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Word length distribution
        ax1.hist(word_lengths, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Word Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Word Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Character length distribution
        ax2.hist(char_lengths, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('Character Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Character Length Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_generation_metrics(
        self, 
        metrics: Dict[str, float], 
        title: str = "Generation Metrics",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot generation metrics as a bar chart.
        
        Args:
            metrics: Dictionary of metrics
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Filter numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        
        if not numeric_metrics:
            print("No numeric metrics to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(numeric_metrics.keys(), numeric_metrics.values())
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Value')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, numeric_metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prompt_vs_generated(
        self, 
        prompts: List[str], 
        generated_texts: List[str],
        title: str = "Prompt vs Generated Text Length",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison between prompt and generated text lengths.
        
        Args:
            prompts: List of input prompts
            generated_texts: List of generated texts
            title: Plot title
            save_path: Optional path to save the plot
        """
        prompt_lengths = [len(prompt.split()) for prompt in prompts]
        generated_lengths = [len(text.split()) for text in generated_texts]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(prompts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prompt_lengths, width, label='Prompt Length', alpha=0.8)
        bars2 = ax.bar(x + width/2, generated_lengths, width, label='Generated Length', alpha=0.8)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Word Count')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_vocabulary_diversity(
        self, 
        texts: List[str], 
        title: str = "Vocabulary Diversity Analysis",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot vocabulary diversity metrics.
        
        Args:
            texts: List of generated texts
            title: Plot title
            save_path: Optional path to save the plot
        """
        diversity_metrics = []
        
        for i, text in enumerate(texts):
            words = text.split()
            unique_words = set(words)
            diversity = len(unique_words) / len(words) if words else 0
            
            diversity_metrics.append({
                'sample': i + 1,
                'total_words': len(words),
                'unique_words': len(unique_words),
                'diversity': diversity
            })
        
        df = pd.DataFrame(diversity_metrics)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Diversity score
        ax1.plot(df['sample'], df['diversity'], marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Diversity Score')
        ax1.set_title('Vocabulary Diversity by Sample')
        ax1.grid(True, alpha=0.3)
        
        # Word count comparison
        ax2.scatter(df['total_words'], df['unique_words'], alpha=0.7, s=60)
        ax2.set_xlabel('Total Words')
        ax2.set_ylabel('Unique Words')
        ax2.set_title('Total vs Unique Words')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['total_words'], df['unique_words'], 1)
        p = np.poly1d(z)
        ax2.plot(df['total_words'], p(df['total_words']), "r--", alpha=0.8)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(
        self, 
        texts: List[str], 
        prompts: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        output_dir: str = "outputs"
    ) -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            texts: List of generated texts
            prompts: Optional list of input prompts
            metrics: Optional dictionary of metrics
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating comprehensive visualization report...")
        
        # Text length distribution
        self.plot_text_length_distribution(
            texts, 
            save_path=str(output_path / "text_length_distribution.png")
        )
        
        # Vocabulary diversity
        self.plot_vocabulary_diversity(
            texts,
            save_path=str(output_path / "vocabulary_diversity.png")
        )
        
        # Prompt vs generated comparison
        if prompts:
            self.plot_prompt_vs_generated(
                prompts, texts,
                save_path=str(output_path / "prompt_vs_generated.png")
            )
        
        # Metrics visualization
        if metrics:
            self.plot_generation_metrics(
                metrics,
                save_path=str(output_path / "generation_metrics.png")
            )
        
        print(f"Visualization report saved to {output_path}")


def visualize_from_file(
    input_file: str, 
    output_dir: str = "outputs"
) -> None:
    """
    Create visualizations from a results file.
    
    Args:
        input_file: Path to JSON file with results
        output_dir: Directory to save visualizations
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        texts = data
        prompts = None
        metrics = None
    elif isinstance(data, dict):
        texts = data.get('generated_texts', [])
        prompts = data.get('prompts', None)
        metrics = data.get('metrics', None)
    else:
        raise ValueError("Invalid file format")
    
    visualizer = TextVisualizer()
    visualizer.create_comprehensive_report(texts, prompts, metrics, output_dir)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a sample generated text for testing purposes.",
        "Another example of generated content with different characteristics.",
        "A third sample to demonstrate the visualization capabilities.",
        "This text has more words and demonstrates longer generation patterns.",
        "Short text."
    ]
    
    visualizer = TextVisualizer()
    visualizer.create_comprehensive_report(sample_texts)
