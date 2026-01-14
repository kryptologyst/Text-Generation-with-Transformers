#!/usr/bin/env python3
"""
Example script demonstrating the modernized text generation capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from text_generator import TextGenerator, GenerationConfig, create_synthetic_dataset
from config import ConfigManager
from logging_config import setup_logging, get_logger
from visualization import TextVisualizer


def main():
    """Main example function."""
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    logger.info("Starting text generation example")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    logger.info(f"Using model: {config.model.name}")
    
    # Initialize generator
    generator = TextGenerator(
        model_name=config.model.name,
        device=config.model.device,
        use_pipeline=config.model.use_pipeline
    )
    
    # Example 1: Single text generation
    logger.info("Example 1: Single text generation")
    prompt = "Once upon a time, in a magical forest"
    
    generation_config = GenerationConfig(
        max_length=150,
        temperature=0.8,
        num_return_sequences=1
    )
    
    generated_text = generator.generate_text(prompt, generation_config)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Example 2: Multiple variations
    logger.info("Example 2: Multiple variations")
    generation_config.num_return_sequences = 3
    generation_config.temperature = 0.9
    
    variations = generator.generate_text(prompt, generation_config)
    print(f"\nPrompt: {prompt}")
    for i, variation in enumerate(variations, 1):
        print(f"Variation {i}: {variation}")
    
    # Example 3: Batch processing
    logger.info("Example 3: Batch processing")
    prompts = [
        "The future of artificial intelligence",
        "In a world where robots",
        "Technology has transformed",
        "The secret to happiness",
        "Dreams can come true"
    ]
    
    batch_config = GenerationConfig(
        max_length=100,
        temperature=0.7,
        num_return_sequences=1
    )
    
    batch_results = generator.generate_multiple_prompts(prompts, batch_config)
    
    print("\nBatch Results:")
    for prompt, result in zip(prompts, batch_results):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result}")
    
    # Example 4: Evaluation
    logger.info("Example 4: Evaluation")
    metrics = generator.evaluate_generation(batch_results)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Example 5: Synthetic dataset
    logger.info("Example 5: Synthetic dataset")
    dataset = create_synthetic_dataset(num_samples=5)
    
    print("\nSynthetic Dataset:")
    for sample in dataset:
        print(f"ID: {sample['id']}, Prompt: {sample['prompt']}, Category: {sample['category']}")
    
    # Example 6: Visualization
    logger.info("Example 6: Visualization")
    try:
        visualizer = TextVisualizer()
        visualizer.plot_text_length_distribution(batch_results, "Batch Results Length Distribution")
        visualizer.plot_generation_metrics(metrics, "Generation Metrics")
    except ImportError:
        logger.warning("Visualization libraries not available. Skipping plots.")
    
    # Example 7: Save results
    logger.info("Example 7: Save results")
    generator.save_generated_texts(
        batch_results,
        "outputs/example_results.json",
        metadata={
            "prompts": prompts,
            "metrics": metrics,
            "model": config.model.name
        }
    )
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
