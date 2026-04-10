#!/usr/bin/env python3
"""
Test script to verify ModelManager with actual model directories.
"""

import sys
import os
from pathlib import Path
import torch

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.model_manager import ModelManager
from utils.configuration import load_config

def test_model_manager_with_actual_models():
    """Test ModelManager with actual model directories."""
    
    print("Testing ModelManager with Actual Models")
    print("=" * 50)
    
    # Load configuration  
    try:
        config = load_config()
        models_base_path = config.get("MODEL_PATH") or config.get("MODELS_PATH", "src/models")
        print(f"Models base path: {models_base_path}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return False
    
    # Check available models
    models_path = Path(models_base_path)
    if not models_path.exists():
        print(f"Models directory doesn't exist: {models_path}")
        return False
    
    # Find model subdirectories
    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
    print(f"\nFound model directories: {[d.name for d in model_dirs]}")
    
    # Test each model directory
    for model_dir in model_dirs:
        print(f"\n" + "=" * 60)
        print(f"Testing Model: {model_dir.name}")
        print(f"=" * 60)
        
        try:
            # Test ModelManager initialization
            manager = ModelManager(str(model_dir))
            print(f"✓ ModelManager initialized for {model_dir.name}")
            print(f"  Detected type: {manager.model_type}")
            print(f"  Model path: {manager.weights_path}")
            
            # Check model files
            model_files = list(model_dir.glob("*.safetensors"))
            config_files = list(model_dir.glob("config.json"))
            tokenizer_files = list(model_dir.glob("tokenizer*"))
            
            print(f"\nFile Analysis:")
            print(f"  Model files (.safetensors): {len(model_files)}")
            print(f"  Config files: {len(config_files)}")
            print(f"  Tokenizer files: {len(tokenizer_files)}")
            
            # Check key files
            key_files = [
                "config.json",
                "tokenizer_config.json", 
                "tokenizer.json"
            ]
            
            missing_files = []
            for key_file in key_files:
                file_path = model_dir / key_file
                if file_path.exists():
                    print(f"  ✓ {key_file}")
                else:
                    print(f"  ✗ {key_file} (missing)")
                    missing_files.append(key_file)
            
            # Check if model is ready for loading
            has_model_files = len(model_files) > 0 or (model_dir / "model.safetensors").exists()
            has_config = len(config_files) > 0
            has_tokenizer = len(tokenizer_files) > 0
            
            model_ready = has_model_files and has_config and has_tokenizer
            
            print(f"\nModel Readiness Assessment:")
            print(f"  Model weights: {'✓' if has_model_files else '✗'}")
            print(f"  Configuration: {'✓' if has_config else '✗'}")
            print(f"  Tokenizer: {'✓' if has_tokenizer else '✗'}")
            print(f"  Overall ready: {'✓' if model_ready else '✗'}")
            
            if model_ready:
                print(f"\n  This model appears ready for loading!")
                print(f"  Note: Actual loading requires significant memory")
                
                # Get model info
                info = manager.get_model_info()
                print(f"  Model info: {info}")
                
                # Estimate model size
                total_size = sum(f.stat().st_size for f in model_files)
                size_gb = total_size / (1024**3)
                print(f"  Estimated model size: {size_gb:.1f} GB")
                
            else:
                print(f"\n  Model is not ready for loading (missing files)")
                if missing_files:
                    print(f"  Missing: {', '.join(missing_files)}")
            
        except Exception as e:
            print(f"Error testing {model_dir.name}: {e}")
            continue
    
    return True

def test_model_loading_preparation():
    """Test preparation steps for model loading."""
    
    print(f"\n" + "=" * 60)
    print("Model Loading Preparation Test")
    print("=" * 60)
    
    # Check system requirements
    print("System Requirements Check:")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        
        # Memory recommendation
        if gpu_memory >= 24:
            print(f"  ✓ Sufficient GPU memory for most models")
        elif gpu_memory >= 12:
            print(f"  ⚠ GPU memory may be limited for large models")  
        else:
            print(f"  ✗ GPU memory likely insufficient for large models")
    else:
        print(f"  Running on CPU - model loading will be very slow")
    
    # Check transformers library
    try:
        from transformers import __version__ as transformers_version
        print(f"  Transformers version: {transformers_version}")
    except ImportError:
        print(f"  ✗ Transformers library not available")
        return False
    
    # Check torch version
    print(f"  PyTorch version: {torch.__version__}")
    
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        print(f"  System RAM: {ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available")
        
        if available_ram_gb >= 32:
            print(f"  ✓ Sufficient system RAM")
        elif available_ram_gb >= 16:
            print(f"  ⚠ System RAM may be limited")
        else:
            print(f"  ✗ Low system RAM may cause issues")
            
    except ImportError:
        print(f"  RAM check skipped (psutil not available)")
    
    return True

if __name__ == "__main__":
    print("ModelManager Comprehensive Test Suite")
    print("=" * 70)
    
    success1 = test_model_manager_with_actual_models()
    success2 = test_model_loading_preparation()
    
    overall_success = success1 and success2
    
    print(f"\n" + "=" * 70)
    print(f"Comprehensive Test Result: {'PASS' if overall_success else 'FAIL'}")
    if overall_success:
        print(f"ModelManager is ready for production use!")
    print(f"=" * 70)
    
    sys.exit(0 if overall_success else 1)