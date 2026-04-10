#!/usr/bin/env python3
"""
Cluster-specific test script for LLM-Needs-a-Plan
Designed to run on Leonardo cluster with proper resource management.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add src to path and get project root
if 'LLM_PROJECT_ROOT' in os.environ:
    project_root = Path(os.environ['LLM_PROJECT_ROOT'])
else:
    # Fallback: go up from src/tests/ to project root
    project_root = Path(__file__).parent.parent.parent

sys.path.insert(0, str(project_root / "src"))

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def test_environment():
    """Test basic environment setup."""
    print_section("ENVIRONMENT CHECK")
    
    print_subsection("Python Environment")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root (from env): {os.environ.get('LLM_PROJECT_ROOT', 'NOT SET')}")
    print(f"Project root (calculated): {project_root}")
    print(f"Data path should be: {project_root / 'src' / 'data'}")
    
    print_subsection("System Information")
    try:
        import platform
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"Architecture: {platform.architecture()}")
    except Exception as e:
        print(f"Could not get system info: {e}")
    
    print_subsection("GPU Information")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("No CUDA GPUs available")
    except ImportError:
        print("PyTorch not available")
    except Exception as e:
        print(f"GPU check failed: {e}")

def test_dependencies():
    """Test required dependencies."""
    print_section("DEPENDENCY CHECK")
    
    required_packages = [
        'torch', 'transformers', 'accelerate', 'sentencepiece', 
        'huggingface_hub', 'safetensors', 'yaml'
    ]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: NOT FOUND")
        except Exception as e:
            print(f"? {package}: ERROR - {e}")

def test_configuration():
    """Test configuration loading."""
    print_section("CONFIGURATION TEST")
    
    try:
        from utils.configuration import load_config
        config = load_config()
        
        print("Configuration loaded successfully:")
        for key, value in config.items():
            if isinstance(value, str) and len(value) > 50:
                value = value[:47] + "..."
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_file_manager():
    """Test file manager functionality."""
    print_section("FILE MANAGER TEST")
    
    try:
        from core.file_manager import FileManager
        
        data_path = project_root / "src" / "data"
        print(f"Looking for data at: {data_path}")
        if data_path.exists():
            fm = FileManager()
            domains = fm.find_pddl_files(str(data_path))
            
            print(f"Found {len(domains)} domains:")
            for domain_info in domains:
                domain_name = getattr(domain_info, 'domain_name', 'unknown')
                problem_paths = getattr(domain_info, 'problem_paths', [])
                problem_count = len(problem_paths)
                print(f"  {domain_name}: {problem_count} problems")
            
            return len(domains) > 0
        else:
            print(f"Data directory not found: {data_path}")
            return False
            
    except Exception as e:
        print(f"File manager test failed: {e}")
        traceback.print_exc()
        return False

def test_val_integration():
    """Test VAL validator integration."""
    print_section("VAL VALIDATOR TEST")
    
    try:
        from utils.validator import get_val_executable, validate_plan_from_text
        
        print_subsection("VAL Executable Check")
        val_path = get_val_executable()
        print(f"VAL executable path: {val_path}")
        
        val_file = Path(val_path)
        if val_file.exists():
            print(f"✓ VAL executable found")
            print(f"  File size: {val_file.stat().st_size} bytes")
            print(f"  Executable: {os.access(val_path, os.X_OK)}")
        else:
            print(f"✗ VAL executable not found")
            return False
        
        print_subsection("VAL Function Test")
        # Test with dummy data (will fail validation but test the interface)
        dummy_domain = "(define (domain test))"
        dummy_problem = "(define (problem test-prob))"
        dummy_plan = "(action1 param1)"
        
        try:
            result = validate_plan_from_text(dummy_domain, dummy_problem, dummy_plan)
            print(f"✓ VAL validation interface working")
            print(f"  Result type: {type(result)}")
            print(f"  Has 'valid' key: {'valid' in result}")
            print(f"  Has 'error' key: {'error' in result}")
            return True
        except Exception as val_e:
            print(f"✗ VAL validation failed: {val_e}")
            return False
            
    except Exception as e:
        print(f"VAL test failed: {e}")
        traceback.print_exc()
        return False

def test_model_detection():
    """Test model detection without loading."""
    print_section("MODEL DETECTION TEST")
    
    try:
        from core.model_manager import ModelManager
        
        # Test model detection for available paths
        model_paths = [
            project_root / "src" / "models" / "Llama3",
            project_root / "src" / "models" / "Phi4",
            project_root / "src" / "models" / "Gemma3"
        ]
        
        models_found = False
        for model_path in model_paths:
            if model_path.exists():
                print_subsection(f"Checking {model_path.name}")
                try:
                    # Create manager with the model path
                    manager = ModelManager(str(model_path))
                    model_type = manager.model_type
                    print(f"✓ Detected model type: {model_type}")
                    models_found = True
                    
                    # List some model files
                    model_files = list(model_path.glob("*.json"))[:3]
                    if model_files:
                        print("  Configuration files found:")
                        for f in model_files:
                            print(f"    {f.name}")
                    
                    # Check model size
                    safetensors_files = list(model_path.glob("*.safetensors"))
                    if safetensors_files:
                        total_size = sum(f.stat().st_size for f in safetensors_files[:])  # All files
                        print(f"  Model size (all files): {total_size / 1e9:.1f} GB")

                except Exception as model_e:
                    print(f"✗ Model detection failed: {model_e}")
            else:
                print(f"Model path not found: {model_path}")
        
        if not models_found:
            print("No models found, but ModelManager class is available")
            # Just test that we can import it
            from core.model_manager import ModelManager
            print("✓ ModelManager class imported successfully")
        
        return True
        
    except Exception as e:
        print(f"Model detection test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all cluster tests."""
    print_section("CLUSTER TEST SUITE")
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Environment Check", test_environment),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("File Manager", test_file_manager),
        ("VAL Integration", test_val_integration),
        ("Model Detection", test_model_detection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'success': result if result is not None else True,
                'time': end_time - start_time
            }
            
        except Exception as e:
            print(f"TEST FAILED: {test_name} - {e}")
            results[test_name] = {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    # Print summary
    print_section("TEST RESULTS SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"Tests completed: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        time_str = f"{result['time']:.2f}s"
        print(f"  {status} {test_name} ({time_str})")
        
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    print(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)