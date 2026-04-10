"""
Common utility functions for the LLM-Needs-a-Plan project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML file.
    
    Args:
        file_path (Path): Path to the YAML file
        
    Returns:
        Dict[str, Any]: Parsed YAML content as a dictionary
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def save_yaml_file(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data (Dict[str, Any]): Data to save
        file_path (Path): Path where to save the YAML file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, indent=2)


def ensure_directory_exists(directory: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (Path): Directory path to check/create
    """
    directory.mkdir(parents=True, exist_ok=True)