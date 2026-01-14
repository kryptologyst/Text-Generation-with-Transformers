# Text Generation with Transformers

A production-ready implementation of text generation using Hugging Face transformers. This project demonstrates state-of-the-art techniques for generating human-like text with various transformer models.

## Features

- **Multiple Models**: Support for GPT-2 variants (small, medium, large) and DistilGPT-2
- **Flexible Interfaces**: CLI, Web UI (Streamlit), and Python API
- **Advanced Generation**: Configurable parameters (temperature, top-p, top-k, etc.)
- **Batch Processing**: Generate text for multiple prompts efficiently
- **Evaluation Metrics**: Comprehensive text quality assessment
- **Modern Architecture**: Type hints, configuration management, logging
- **Production Ready**: Error handling, caching, and performance optimization

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Text-Generation-with-Transformers.git
cd Text-Generation-with-Transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

Generate text from a single prompt:
```bash
python cli.py generate --prompt "Once upon a time, in a magical forest"
```

Generate multiple variations:
```bash
python cli.py generate --prompt "The future of AI" --num-sequences 3 --temperature 0.8
```

Batch processing:
```bash
python cli.py batch --input prompts.json --output results.json
```

#### Web Interface

Launch the interactive web application:
```bash
python cli.py web
# or
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

#### Python API

```python
from src.text_generator import TextGenerator, GenerationConfig

# Initialize generator
generator = TextGenerator(model_name="gpt2")

# Generate text
config = GenerationConfig(max_length=200, temperature=0.7)
text = generator.generate_text("Once upon a time", config)
print(text)
```

## 📁 Project Structure

```
0530_Text_Generation_with_Transformers/
├── src/                    # Core source code
│   ├── text_generator.py   # Main text generation class
│   └── config.py          # Configuration management
├── web_app/               # Streamlit web interface
│   └── app.py             # Web application
├── data/                  # Data directory
├── models/                # Model storage
├── outputs/               # Generated outputs
├── config/                # Configuration files
├── tests/                 # Test files
├── cli.py                 # Command-line interface
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Configuration

### Model Configuration

Available models:
- `gpt2`: Fast, good for experimentation
- `gpt2-medium`: Balanced performance and speed
- `gpt2-large`: Higher quality, slower generation
- `distilgpt2`: Lightweight version

### Generation Parameters

- **Temperature** (0.1-2.0): Controls randomness. Lower values = more focused
- **Top-p** (0.1-1.0): Nucleus sampling threshold
- **Top-k** (1-100): Limit vocabulary to top-k tokens
- **Max Length**: Maximum length of generated text
- **Num Sequences**: Number of different texts to generate

### Configuration Files

Create a configuration file (`config/config.yaml`):

```yaml
model:
  name: "gpt2"
  device: "auto"
  use_pipeline: true

generation:
  max_length: 200
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  num_return_sequences: 1

data_dir: "data"
output_dir: "outputs"
log_level: "INFO"
random_seed: 42
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Average Length**: Mean word count of generated texts
- **Vocabulary Diversity**: Ratio of unique words to total words
- **Perplexity**: Model's confidence in generated text
- **Character Count**: Average character length

## 🔧 Advanced Usage

### Custom Models

Use any Hugging Face causal language model:

```python
generator = TextGenerator(model_name="microsoft/DialoGPT-medium")
```

### Batch Processing

Process multiple prompts efficiently:

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = generator.generate_multiple_prompts(prompts)
```

### Evaluation

Evaluate generated texts:

```python
metrics = generator.evaluate_generation(generated_texts)
print(f"Average length: {metrics['avg_length']:.2f} words")
```

### Synthetic Dataset

Create synthetic data for testing:

```python
from src.text_generator import create_synthetic_dataset

dataset = create_synthetic_dataset(num_samples=100)
```

## Testing

Run tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Deployment

### Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "web_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t text-generator .
docker run -p 8501:8501 text-generator
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster generation
2. **Model Caching**: Models are automatically cached by Hugging Face
3. **Batch Processing**: Process multiple prompts together for efficiency
4. **Pipeline Mode**: Use pipeline mode for simpler, optimized inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [OpenAI](https://openai.com/) for GPT-2 models
- [Streamlit](https://streamlit.io/) for the web interface

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Issues**: Check internet connection and disk space
3. **Import Errors**: Ensure all dependencies are installed
# Text-Generation-with-Transformers
