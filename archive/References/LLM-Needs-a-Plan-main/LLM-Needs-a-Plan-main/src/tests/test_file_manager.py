#!/usr/bin/env python3
"""
Test script to verify the PDDL data structure and the file manager works accordingly.
"""

# Standard libraries
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Local imports
from core.file_manager import FileManager
from utils.configuration import load_config

def main():
    """Test the new PDDL directory structure."""
    
    print("Testing PDDL Data Structure")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        problems_path = config["PROBLEMS_PATH"]
        print(f"Problems path from config: {problems_path}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return False
        
    # Test directory structure
    abs_problems_path = Path(__file__).parent.parent.parent / problems_path
    print(f"Absolute problems path: {abs_problems_path}")
    
    if not abs_problems_path.exists():
        print(f"Problems directory doesn't exist: {abs_problems_path}")
        return False
    
    # List domain directories
    print(f"\nDomain directories found:")
    domain_dirs = []
    for item in abs_problems_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            domain_dirs.append(item.name)
            print(f"  - {item.name}")
    
    if not domain_dirs:
        print("No domain directories found")
        print("Expected structure:")
        print("  src/data/")
        print("  ├── tetris/")
        print("  ├── citycar/")
        print("  └── logistics/")
        return False

    # Initialize file manager
    file_manager = FileManager()
    
    # Test file manager
    print(f"\nTesting FileManager.find_pddl_files()...")
    try:
        pddl_data = file_manager.find_pddl_files(str(abs_problems_path))

        if not pddl_data:
            print("No PDDL files found (this is expected if no domain/problem files are added yet)")
            print("To test fully, add PDDL files to domain directories:")
            for domain in domain_dirs:
                print(f"  {problems_path}/{domain}/domain.pddl")
                print(f"  {problems_path}/{domain}/problem_01.pddl")
        else:
            print(f"Found {len(pddl_data)} domains:")
            for domain_data in pddl_data:
                domain_name = domain_data.domain_name
                problem_count = len(domain_data.problem_paths)
                print(f"  - {domain_name}: {problem_count} problems")
                
    except Exception as e:
        print(f"FileManager error: {e}")
        return False
    
    print(f"\nDirectory structure test completed!")    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)