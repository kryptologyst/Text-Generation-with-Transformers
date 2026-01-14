"""
Configuration management for the text generation project.
"""

import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "gpt2"
    device: str = "auto"
    use_pipeline: bool = True
    cache_dir: Optional[str] = None


@dataclass
class GenerationConfig:
    """Text generation configuration parameters."""
    max_length: int = 200
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    no_repeat_ngram_size: int = 2
    do_sample: bool = True


@dataclass
class AppConfig:
    """Application configuration."""
    model: ModelConfig
    generation: GenerationConfig
    data_dir: str = "data"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    random_seed: int = 42


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_default_config()
        
        if Path(self.config_path).exists():
            self.load_config(self.config_path)
    
    def _load_default_config(self) -> AppConfig:
        """Load default configuration."""
        return AppConfig(
            model=ModelConfig(),
            generation=GenerationConfig()
        )
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update configuration
        if 'model' in config_data:
            model_config = ModelConfig(**config_data['model'])
        else:
            model_config = ModelConfig()
        
        if 'generation' in config_data:
            generation_config = GenerationConfig(**config_data['generation'])
        else:
            generation_config = GenerationConfig()
        
        self.config = AppConfig(
            model=model_config,
            generation=generation_config,
            data_dir=config_data.get('data_dir', 'data'),
            output_dir=config_data.get('output_dir', 'outputs'),
            log_level=config_data.get('log_level', 'INFO'),
            random_seed=config_data.get('random_seed', 42)
        )
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


def create_default_config_file(config_path: str = "config/config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save the default configuration
    """
    config_manager = ConfigManager()
    config_manager.save_config(config_path)
    print(f"Default configuration saved to {config_path}")


if __name__ == "__main__":
    create_default_config_file()
