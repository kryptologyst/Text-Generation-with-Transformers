#!/usr/bin/env python3
"""
Simple test script to verify the project structure and imports work correctly.
This script tests the code without requiring model downloads.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from config import ConfigManager, ModelConfig, GenerationConfig
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Config module import failed: {e}")
        return False
    
    try:
        from logging_config import setup_logging, get_logger
        print("✓ Logging config module imported successfully")
    except ImportError as e:
        print(f"✗ Logging config module import failed: {e}")
        return False
    
    try:
        from visualization import TextVisualizer
        print("✓ Visualization module imported successfully")
    except ImportError as e:
        print(f"✗ Visualization module import failed: {e}")
        return False
    
    # Test text_generator imports (without initializing the model)
    try:
        from text_generator import GenerationConfig as TextGenConfig, create_synthetic_dataset
        print("✓ Text generator module imported successfully")
    except ImportError as e:
        print(f"✗ Text generator module import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration functionality."""
    print("\nTesting configuration...")
    
    try:
        from config import ConfigManager
        
        # Test default config
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert config.model.name == "gpt2"
        assert config.generation.max_length == 200
        print("✓ Default configuration works correctly")
        
        # Test config update
        config_manager.update_config(data_dir="test_data")
        updated_config = config_manager.get_config()
        assert updated_config.data_dir == "test_data"
        print("✓ Configuration update works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_synthetic_dataset():
    """Test synthetic dataset creation."""
    print("\nTesting synthetic dataset...")
    
    try:
        from text_generator import create_synthetic_dataset
        
        dataset = create_synthetic_dataset(num_samples=5)
        
        assert len(dataset) == 5
        assert all("id" in sample for sample in dataset)
        assert all("prompt" in sample for sample in dataset)
        assert all("category" in sample for sample in dataset)
        
        print("✓ Synthetic dataset creation works correctly")
        return True
    except Exception as e:
        print(f"✗ Synthetic dataset test failed: {e}")
        return False

def test_logging():
    """Test logging configuration."""
    print("\nTesting logging...")
    
    try:
        from logging_config import setup_logging, get_logger
        
        setup_logging(level="INFO")
        logger = get_logger("test")
        
        logger.info("Test log message")
        print("✓ Logging configuration works correctly")
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "src/text_generator.py",
        "src/config.py",
        "src/logging_config.py",
        "src/visualization.py",
        "web_app/app.py",
        "cli.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "config/config.yaml",
        "data/sample_prompts.json",
        "tests/test_text_generator.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("PROJECT STRUCTURE AND IMPORT TESTS")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_synthetic_dataset,
        test_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the example: python example.py")
        print("3. Launch web interface: python cli.py web")
        print("4. Run tests: pytest tests/")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
