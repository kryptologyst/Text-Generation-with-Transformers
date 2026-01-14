"""
Tests for the text generation module.
"""

import pytest
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from text_generator import TextGenerator, GenerationConfig, create_synthetic_dataset
from config import ConfigManager, ModelConfig, GenerationConfig as ConfigGenerationConfig


class TestTextGenerator:
    """Test cases for TextGenerator class."""
    
    @patch('text_generator.pipeline')
    def test_init_with_pipeline(self, mock_pipeline):
        """Test TextGenerator initialization with pipeline."""
        mock_pipeline.return_value = Mock()
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        
        assert generator.model_name == "gpt2"
        assert generator.use_pipeline is True
        mock_pipeline.assert_called_once()
    
    @patch('text_generator.AutoTokenizer')
    @patch('text_generator.AutoModelForCausalLM')
    def test_init_without_pipeline(self, mock_model, mock_tokenizer):
        """Test TextGenerator initialization without pipeline."""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=False)
        
        assert generator.model_name == "gpt2"
        assert generator.use_pipeline is False
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    @patch('text_generator.pipeline')
    def test_generate_text_single(self, mock_pipeline):
        """Test single text generation."""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "This is generated text."}]
        mock_pipeline.return_value = mock_generator
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        config = GenerationConfig(num_return_sequences=1)
        
        result = generator.generate_text("Test prompt", config)
        
        assert result == "This is generated text."
        mock_generator.assert_called_once()
    
    @patch('text_generator.pipeline')
    def test_generate_text_multiple(self, mock_pipeline):
        """Test multiple text generation."""
        mock_generator = Mock()
        mock_generator.return_value = [
            {"generated_text": "Text 1"},
            {"generated_text": "Text 2"}
        ]
        mock_pipeline.return_value = mock_generator
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        config = GenerationConfig(num_return_sequences=2)
        
        result = generator.generate_text("Test prompt", config)
        
        assert result == ["Text 1", "Text 2"]
    
    @patch('text_generator.pipeline')
    def test_generate_multiple_prompts(self, mock_pipeline):
        """Test generation for multiple prompts."""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Generated text"}]
        mock_pipeline.return_value = mock_generator
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        prompts = ["Prompt 1", "Prompt 2"]
        
        results = generator.generate_multiple_prompts(prompts)
        
        assert len(results) == 2
        assert all(result == "Generated text" for result in results)
        assert mock_generator.call_count == 2
    
    @patch('text_generator.pipeline')
    def test_evaluate_generation(self, mock_pipeline):
        """Test text evaluation."""
        mock_pipeline.return_value = Mock()
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        generated_texts = ["This is a test.", "Another test text."]
        
        metrics = generator.evaluate_generation(generated_texts)
        
        assert "avg_length" in metrics
        assert "avg_chars" in metrics
        assert "vocabulary_diversity" in metrics
        assert metrics["avg_length"] > 0
        assert metrics["avg_chars"] > 0
    
    @patch('text_generator.pipeline')
    @patch('builtins.open', create=True)
    def test_save_generated_texts(self, mock_open, mock_pipeline):
        """Test saving generated texts."""
        mock_pipeline.return_value = Mock()
        
        generator = TextGenerator(model_name="gpt2", use_pipeline=True)
        texts = ["Text 1", "Text 2"]
        
        generator.save_generated_texts(texts, "test_output.json")
        
        mock_open.assert_called_once_with("test_output.json", 'w', encoding='utf-8')


class TestGenerationConfig:
    """Test cases for GenerationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.max_length == 200
        assert config.num_return_sequences == 1
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.no_repeat_ngram_size == 2
        assert config.do_sample is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_length=100,
            temperature=0.5,
            num_return_sequences=3
        )
        
        assert config.max_length == 100
        assert config.temperature == 0.5
        assert config.num_return_sequences == 3


class TestSyntheticDataset:
    """Test cases for synthetic dataset creation."""
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_samples=10)
        
        assert len(dataset) == 10
        assert all("id" in sample for sample in dataset)
        assert all("prompt" in sample for sample in dataset)
        assert all("category" in sample for sample in dataset)
    
    def test_dataset_content(self):
        """Test synthetic dataset content."""
        dataset = create_synthetic_dataset(num_samples=5)
        
        # Check that prompts are not empty
        assert all(len(sample["prompt"]) > 0 for sample in dataset)
        
        # Check that categories are valid
        valid_categories = {"once", "in", "the", "technology", "love", "success", "happiness", "the", "dreams", "life"}
        assert all(sample["category"] in valid_categories for sample in dataset)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert config.model.name == "gpt2"
        assert config.generation.max_length == 200
        assert config.data_dir == "data"
    
    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_load_yaml_config(self, mock_yaml_load, mock_open):
        """Test loading YAML configuration."""
        mock_yaml_load.return_value = {
            "model": {"name": "gpt2-medium"},
            "generation": {"max_length": 300},
            "data_dir": "custom_data"
        }
        
        config_manager = ConfigManager()
        config_manager.load_config("test_config.yaml")
        config = config_manager.get_config()
        
        assert config.model.name == "gpt2-medium"
        assert config.generation.max_length == 300
        assert config.data_dir == "custom_data"
    
    def test_update_config(self):
        """Test configuration updates."""
        config_manager = ConfigManager()
        config_manager.update_config(data_dir="updated_data")
        
        config = config_manager.get_config()
        assert config.data_dir == "updated_data"
    
    def test_invalid_config_update(self):
        """Test invalid configuration update."""
        config_manager = ConfigManager()
        
        with pytest.raises(ValueError):
            config_manager.update_config(invalid_param="value")


if __name__ == "__main__":
    pytest.main([__file__])
