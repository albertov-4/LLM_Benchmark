#!/usr/bin/env python3
"""
Test script to verify the PDDLProcessor integration with FileManager.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.file_manager import FileManager
from core.pddl_processor import PDDLProcessor
from utils.configuration import load_config

def test_pddl_processor_integration():
    """Test the PDDLProcessor with FileManager integration."""
    
    print("Testing PDDLProcessor Integration")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        problems_path = config["PROBLEMS_PATH"]
        print(f"Problems path from config: {problems_path}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return False
    
    # Initialize file manager and get domains
    file_manager = FileManager()
    
    # Test directory structure
    abs_problems_path = Path(__file__).parent.parent.parent / problems_path
    print(f"Absolute problems path: {abs_problems_path}")
    
    if not abs_problems_path.exists():
        print(f"Problems directory doesn't exist: {abs_problems_path}")
        return False
    
    # Get PDDL domains
    print(f"\nTesting FileManager.find_pddl_files()...")
    try:
        pddl_data = file_manager.find_pddl_files(str(abs_problems_path))
        
        if not pddl_data:
            print("No PDDL files found (need domain files to test PDDLProcessor)")
            return True
        
        print(f"Found {len(pddl_data)} domains:")
        for domain_data in pddl_data:
            domain_name = domain_data.domain_name
            problem_count = len(domain_data.problem_paths)
            print(f"  - {domain_name}: {problem_count} problems")
        
        # Test PDDLProcessor initialization (without model)
        print(f"\nTesting PDDLProcessor initialization...")
        output_dir = "test_output"
        
        # Create a mock model manager for testing
        class MockModelManager:
            def get_model_info(self):
                return {"loaded": False, "mock": True}
            
            def iterative_planning_with_validation(self, **kwargs):
                return "Mock plan response", 1, True
        
        mock_model = MockModelManager()
        processor = PDDLProcessor(model_manager=mock_model, output_dir=output_dir)
        
        print(f"PDDLProcessor created successfully")
        print(f"Output directory: {processor.output_dir}")
        
        # Test processor info
        info = processor.get_processor_info()
        print(f"Processor info: {info}")
        
        print(f"\nPDDLProcessor integration test completed successfully!")
        print(f"Ready for full planning pipeline with loaded models.")
        
    except Exception as e:
        print(f"PDDLProcessor error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_pddl_processor_integration()
    sys.exit(0 if success else 1)