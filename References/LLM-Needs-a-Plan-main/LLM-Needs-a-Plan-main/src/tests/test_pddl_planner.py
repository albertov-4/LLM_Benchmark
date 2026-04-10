#!/usr/bin/env python3
"""
Test script to verify the PDDLPlanner setup and integration.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.pddl_planner import PDDLPlanner
from utils.configuration import load_config

class MockArgs:
    """Mock arguments class for testing."""
    def __init__(self):
        self.problems_path = "src/data"
        self.weights_path = "src/models"
        self.output_dir = "./test_planner_output"
        self.domain = None
        self.batch = False
        self.max_iterations = 3
        self.cot = False
        self.add_system_prompt = True
        self.sampling = False
        self.max_tokens = 5000
        self.temperature = 0.6
        self.top_k = 10
        self.include_prompt = False
        self.skip_special_tokens = True
        self.model = "auto"
        self.verbose = False
        self.log_level = "INFO"
        self.log_file = None

def test_pddl_planner_setup():
    """Test PDDLPlanner setup without model loading."""
    
    print("Testing PDDLPlanner Setup")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        print(f"Configuration loaded successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        return False
    
    # Create mock arguments
    args = MockArgs()
    
    # Test PDDLPlanner initialization
    try:
        planner = PDDLPlanner(args, config)
        print(f"PDDLPlanner initialized successfully")
        
        # Get planner info before setup
        info = planner.get_planner_info()
        print(f"Planner info (before setup):")
        print(f"  Domains available: {info['domains_available']}")
        print(f"  Model info: {info['model_info']}")
        
    except Exception as e:
        print(f"Error initializing PDDLPlanner: {e}")
        return False
    
    # Test file discovery (partial setup)
    try:
        print(f"\nTesting file discovery...")
        from core.file_manager import FileManager
        
        file_manager = FileManager()
        domains_data = file_manager.find_pddl_files(args.problems_path)
        
        if domains_data:
            print(f"Found {len(domains_data)} domain(s):")
            for domain_data in domains_data:
                domain_name = domain_data.domain_name
                problem_count = len(domain_data.problem_paths)
                print(f"  - {domain_name}: {problem_count} problems")
        else:
            print("No domains found")
        
        # Update planner with discovered domains
        planner.domains_data = domains_data
        planner.file_manager = file_manager
        
        # Test domain filtering
        if domains_data:
            first_domain = domains_data[0].domain_name
            print(f"\nTesting domain filtering for: {first_domain}")
            args.domain = first_domain
            
            filtered_domains = [
                d for d in domains_data 
                if d.domain_name.lower() == args.domain.lower()
            ]
            print(f"Filtered to {len(filtered_domains)} domain(s)")
            
    except Exception as e:
        print(f"Error during file discovery: {e}")
        return False
    
    # Test path resolution
    try:
        print(f"\nTesting model path resolution...")
        model_path = planner._resolve_model_path()
        print(f"Resolved model path: {model_path}")
        
        if Path(model_path).exists():
            config_exists = (Path(model_path) / "config.json").exists()
            print(f"Model directory exists: True")
            print(f"Config file exists: {config_exists}")
        else:
            print(f"Model directory exists: False")
        
    except Exception as e:
        print(f"Error resolving model path: {e}")
        return False
    
    # Clean up test directory
    try:
        import shutil
        if Path(args.output_dir).exists():
            shutil.rmtree(args.output_dir)
    except:
        pass
    
    print(f"\nPDDLPlanner setup test completed successfully!")
    print(f"Note: Model loading test skipped (requires significant memory)")
    
    return True

def test_main_integration():
    """Test integration with main.py structure."""
    
    print(f"\n" + "=" * 50)
    print("Testing Main.py Integration")
    print("=" * 50)
    
    # Test that all imports work
    try:
        from core import PDDLPlanner, FileManager, ModelManager, PDDLProcessor
        print("All core modules imported successfully")
        
        # Test argument structure compatibility
        args = MockArgs()
        planner = PDDLPlanner(args)
        print("PDDLPlanner accepts argument structure")
        
        # Test method availability
        methods = ['setup', 'run', 'get_results', 'get_planner_info']
        for method in methods:
            if hasattr(planner, method):
                print(f"  Method '{method}' available")
            else:
                print(f"  Method '{method}' missing")
                return False
        
    except Exception as e:
        print(f"Integration test error: {e}")
        return False
    
    print("Main.py integration test completed successfully!")
    return True

if __name__ == "__main__":
    print("PDDLPlanner Test Suite")
    print("=" * 60)
    
    success1 = test_pddl_planner_setup()
    success2 = test_main_integration()
    
    overall_success = success1 and success2
    
    print(f"\n" + "=" * 60)
    print(f"Overall Test Result: {'PASS' if overall_success else 'FAIL'}")
    if overall_success:
        print("PDDLPlanner is ready for use with main.py!")
        print("To run with actual models, use: python3 src/main.py")
    print("=" * 60)
    
    sys.exit(0 if overall_success else 1)