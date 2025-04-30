import os
import sys

def test_project_paths():
    """Test different project root paths and their contents"""
    
    # Project root (one level up from tests directory)
    path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("\nTesting Project Root:")
    print(f"Path: {path1}")
    print(f"Exists: {os.path.exists(path1)}")
    if os.path.exists(path1):
        print("Contents:")
        for item in os.listdir(path1):
            print(f"  - {item}")
    
    # Check for required module directories
    required_dirs = ['data_processing', 'model_training', 'evaluation']
    print("\nChecking for required directories:")
    
    for dir_name in required_dirs:
        exists = os.path.exists(os.path.join(path1, dir_name))
        print(f"  - {dir_name}: {'✓' if exists else '✗'}")

if __name__ == "__main__":
    test_project_paths()