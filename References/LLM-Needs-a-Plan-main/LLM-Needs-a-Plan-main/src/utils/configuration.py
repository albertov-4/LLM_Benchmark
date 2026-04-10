import sys
from pathlib import Path
from .common_utils import load_yaml_file

# Add the src directory to the Python path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def load_config():
    """
    Load configuration settings from a YAML file located in the root directory.

    :return: Configuration settings as a dictionary.
    """
    # Look for config.yml in the project root
    config_path = Path(__file__).parent.parent.parent.joinpath("config.yml")
    
    if not config_path.exists():
        # Try alternative location in src directory
        config_path = Path(__file__).parent.parent.joinpath("config.yml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    return load_yaml_file(config_path)


def main():
    """Main function when script is run directly."""
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(f"Config file path: {Path(__file__).parent.parent.parent.joinpath('config.yml')}")
        print("Configuration contents:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    return 0


if __name__ == "__main__":
    # When run directly
    exit_code = main()
    sys.exit(exit_code)
else:
    # When imported as a module
    CONFIG = load_config()
