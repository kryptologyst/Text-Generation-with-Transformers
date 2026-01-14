"""
Command-line interface for text generation with transformers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from text_generator import TextGenerator, GenerationConfig, create_synthetic_dataset
from config import ConfigManager


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Text Generation with Transformers CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text from a single prompt
  python cli.py generate --prompt "Once upon a time"
  
  # Generate multiple texts with custom parameters
  python cli.py generate --prompt "The future of AI" --num-sequences 3 --temperature 0.8
  
  # Batch generation from file
  python cli.py batch --input prompts.json --output results.json
  
  # Create synthetic dataset
  python cli.py create-dataset --num-samples 50 --output synthetic_data.json
  
  # Evaluate generated texts
  python cli.py evaluate --input generated_texts.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text from a prompt')
    generate_parser.add_argument('--prompt', '-p', required=True, help='Input prompt')
    generate_parser.add_argument('--model', '-m', default='gpt2', help='Model name')
    generate_parser.add_argument('--device', '-d', default='auto', help='Device to use')
    generate_parser.add_argument('--max-length', type=int, default=200, help='Maximum length')
    generate_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    generate_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p value')
    generate_parser.add_argument('--top-k', type=int, default=50, help='Top-k value')
    generate_parser.add_argument('--num-sequences', type=int, default=1, help='Number of sequences')
    generate_parser.add_argument('--output', '-o', help='Output file')
    generate_parser.add_argument('--config', '-c', help='Configuration file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch text generation')
    batch_parser.add_argument('--input', '-i', required=True, help='Input file with prompts')
    batch_parser.add_argument('--output', '-o', required=True, help='Output file')
    batch_parser.add_argument('--model', '-m', default='gpt2', help='Model name')
    batch_parser.add_argument('--device', '-d', default='auto', help='Device to use')
    batch_parser.add_argument('--max-length', type=int, default=200, help='Maximum length')
    batch_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    batch_parser.add_argument('--config', '-c', help='Configuration file')
    
    # Create dataset command
    dataset_parser = subparsers.add_parser('create-dataset', help='Create synthetic dataset')
    dataset_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    dataset_parser.add_argument('--output', '-o', required=True, help='Output file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate generated texts')
    evaluate_parser.add_argument('--input', '-i', required=True, help='Input file with generated texts')
    evaluate_parser.add_argument('--model', '-m', default='gpt2', help='Model name for evaluation')
    evaluate_parser.add_argument('--output', '-o', help='Output file for metrics')
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=8501, help='Port number')
    web_parser.add_argument('--host', default='localhost', help='Host address')
    
    return parser


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [item.get('prompt', str(item)) for item in data]
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    else:
        raise ValueError("Invalid file format. Expected list of prompts or dict with 'prompts' key.")


def save_results(results: dict, output_path: str) -> None:
    """Save results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def generate_command(args) -> None:
    """Handle generate command."""
    # Load configuration if provided
    config_manager = ConfigManager(args.config) if args.config else ConfigManager()
    
    # Initialize generator
    print(f"Loading model: {args.model}")
    generator = TextGenerator(
        model_name=args.model,
        device=args.device,
        use_pipeline=True
    )
    
    # Create generation config
    config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_sequences
    )
    
    # Generate text
    print(f"Generating text for prompt: '{args.prompt}'")
    generated_text = generator.generate_text(args.prompt, config)
    
    # Display results
    print("\n" + "="*50)
    print("GENERATED TEXT")
    print("="*50)
    
    if args.num_sequences == 1:
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text}")
    else:
        for i, text in enumerate(generated_text):
            print(f"\nSequence {i+1}:")
            print(f"Prompt: {args.prompt}")
            print(f"Generated: {text}")
    
    # Save results if output file specified
    if args.output:
        results = {
            "prompt": args.prompt,
            "generated_texts": generated_text if args.num_sequences > 1 else [generated_text],
            "model": args.model,
            "config": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_sequences": args.num_sequences
            }
        }
        save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")


def batch_command(args) -> None:
    """Handle batch command."""
    # Load prompts
    print(f"Loading prompts from: {args.input}")
    prompts = load_prompts_from_file(args.input)
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize generator
    print(f"Loading model: {args.model}")
    generator = TextGenerator(
        model_name=args.model,
        device=args.device,
        use_pipeline=True
    )
    
    # Create generation config
    config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=1
    )
    
    # Generate texts
    print("Generating texts...")
    generated_texts = generator.generate_multiple_prompts(prompts, config)
    
    # Prepare results
    results = {
        "prompts": prompts,
        "generated_texts": generated_texts,
        "model": args.model,
        "config": {
            "max_length": args.max_length,
            "temperature": args.temperature
        }
    }
    
    # Save results
    save_results(results, args.output)
    print(f"Results saved to: {args.output}")
    
    # Display sample results
    print("\n" + "="*50)
    print("SAMPLE RESULTS")
    print("="*50)
    for i in range(min(3, len(prompts))):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Generated: {generated_texts[i]}")


def create_dataset_command(args) -> None:
    """Handle create-dataset command."""
    print(f"Creating synthetic dataset with {args.num_samples} samples...")
    dataset = create_synthetic_dataset(args.num_samples)
    
    # Convert to list for JSON serialization
    data = [{"id": sample["id"], "prompt": sample["prompt"], "category": sample["category"]} 
            for sample in dataset]
    
    save_results(data, args.output)
    print(f"Dataset saved to: {args.output}")
    
    # Display sample
    print("\n" + "="*50)
    print("SAMPLE DATA")
    print("="*50)
    for i in range(min(5, len(data))):
        print(f"ID: {data[i]['id']}, Prompt: {data[i]['prompt']}, Category: {data[i]['category']}")


def evaluate_command(args) -> None:
    """Handle evaluate command."""
    # Load generated texts
    print(f"Loading generated texts from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        generated_texts = data
    elif isinstance(data, dict) and 'generated_texts' in data:
        generated_texts = data['generated_texts']
    else:
        raise ValueError("Invalid file format for evaluation")
    
    print(f"Loaded {len(generated_texts)} generated texts")
    
    # Initialize generator for evaluation
    generator = TextGenerator(model_name=args.model, use_pipeline=True)
    
    # Evaluate
    print("Computing evaluation metrics...")
    metrics = generator.evaluate_generation(generated_texts)
    
    # Display metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics if output file specified
    if args.output:
        results = {
            "metrics": metrics,
            "num_texts": len(generated_texts),
            "model": args.model
        }
        save_results(results, args.output)
        print(f"\nMetrics saved to: {args.output}")


def web_command(args) -> None:
    """Handle web command."""
    import subprocess
    import os
    
    web_app_path = Path(__file__).parent.parent / "web_app" / "app.py"
    
    print(f"Launching web interface on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            "streamlit", "run", str(web_app_path),
            "--server.port", str(args.port),
            "--server.address", args.host
        ])
    except KeyboardInterrupt:
        print("\nWeb interface stopped.")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with: pip install streamlit")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'generate':
            generate_command(args)
        elif args.command == 'batch':
            batch_command(args)
        elif args.command == 'create-dataset':
            create_dataset_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'web':
            web_command(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
