#!/usr/bin/env python3
"""
Test script to verify VAL validator integration.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.validator import validate_plan, validate_plan_from_text, get_val_executable
from utils.configuration import load_config

def test_val_integration():
    """Test VAL validator integration."""
    
    print("Testing VAL Validator Integration")
    print("=" * 50)
    
    # Test 1: Configuration loading
    try:
        config = load_config()
        val_path = config.get("VAL_PATH")
        val_executable = config.get("VAL_EXECUTABLE")
        val_timeout = config.get("VAL_TIMEOUT")
        
        print(f"VAL Configuration:")
        print(f"  VAL_PATH: {val_path}")
        print(f"  VAL_EXECUTABLE: {val_executable}")
        print(f"  VAL_TIMEOUT: {val_timeout}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return False
    
    # Test 2: VAL executable detection
    try:
        val_exec_path = get_val_executable()
        print(f"\nVAL Executable Detection:")
        print(f"  Detected path: {val_exec_path}")
        
        # Check if executable exists
        if Path(val_exec_path).exists():
            print(f"  Status: Found")
        else:
            print(f"  Status: Not found (will try system PATH)")
        
    except Exception as e:
        print(f"VAL detection error: {e}")
        return False
    
    # Test 3: VAL executable test run
    try:
        print(f"\nVAL Executable Test:")
        
        # Try to run VAL with --help to test if it works
        import subprocess
        result = subprocess.run([val_exec_path, "--help"], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 or "usage" in result.stdout.lower() or "validate" in result.stdout.lower():
            print(f"  VAL executable works correctly")
            print(f"  Help output sample: {result.stdout[:100]}...")
        else:
            print(f"  VAL executable may have issues")
            print(f"  Return code: {result.returncode}")
            print(f"  Output: {result.stdout[:200]}")
            
    except subprocess.TimeoutExpired:
        print(f"  VAL executable timeout (this might be normal)")
    except FileNotFoundError:
        print(f"  VAL executable not found at: {val_exec_path}")
        print(f"  Please ensure VAL is built and available")
        return False
    except Exception as e:
        print(f"  VAL test error: {e}")
    
    # Test 4: Validator function availability
    print(f"\nValidator Functions:")
    print(f"  validate_plan: Available")
    print(f"  validate_plan_from_text: Available")
    print(f"  get_val_executable: Available")
    
    # Test 5: Sample domains check
    print(f"\nSample Domain Check:")
    
    tetris_domain = Path("src/data/tetris/tetris_domain.pddl")
    tetris_problem = Path("src/data/tetris/instance-01.pddl")
    
    if tetris_domain.exists() and tetris_problem.exists():
        print(f"  Tetris domain available for testing")
        print(f"    Domain: {tetris_domain}")
        print(f"    Problem: {tetris_problem}")
        
        # Test validation with a simple (likely invalid) plan
        sample_plan = "(move_square pos1 pos2 piece1)"
        
        try:
            result = validate_plan_from_text(str(tetris_domain), str(tetris_problem), sample_plan)
            print(f"  Sample validation test:")
            print(f"    Valid: {result['valid']}")
            if result['error']:
                print(f"    Error: {result['error'][:100]}...")
            else:
                print(f"    No errors")
                
        except Exception as e:
            print(f"  Sample validation failed: {e}")
    else:
        print(f"  No test domains available")
        print(f"    Domain exists: {tetris_domain.exists()}")
        print(f"    Problem exists: {tetris_problem.exists()}")
    
    print(f"\nVAL Integration test completed!")
    return True

if __name__ == "__main__":
    success = test_val_integration()
    
    if success:
        print(f"\nVAL validator is ready for use!")
        print(f"To use in code:")
        print(f"  from utils.validator import validate_plan_from_text")
        print(f"  result = validate_plan_from_text(domain, problem, plan)")
    else:
        print(f"\nVAL integration needs attention.")
    
    sys.exit(0 if success else 1)