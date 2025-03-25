import os
import sys

def add_project_root():
    """Adds the project root directory to the Python path."""
    # get the directory where this file is located (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # add to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    
    return project_root

# auto-run when imported
project_root = add_project_root()