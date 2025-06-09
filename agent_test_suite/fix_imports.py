#!/usr/bin/env python3
"""
Fix import issues in all test files.
"""

import os
from pathlib import Path

def fix_test_file(file_path):
    """Fix imports and model instantiation in a test file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix imports
    content = content.replace('from aider.models import Models', 'from aider import models')
    
    # Fix model instantiation
    content = content.replace(
        """        # Create models factory and get model
        models_factory = Models(args)
        main_model = models_factory.get_model(args.model)""",
        """        # Create model instance
        main_model = models.Model(
            args.model,
            weak_model=getattr(args, 'weak_model', None),
            verbose=args.verbose
        )"""
    )
    
    # Also fix inline model creation in Python scripts
    content = content.replace(
        """# Create model
models_factory = Models(args)
main_model = models_factory.get_model(args.model)""",
        """# Create model
main_model = models.Model(
    args.model,
    weak_model=getattr(args, 'weak_model', None),
    verbose=args.verbose
)"""
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")


def main():
    """Fix all Python test files."""
    test_dir = Path(__file__).parent / "python_tests"
    
    for test_file in test_dir.glob("test_*.py"):
        fix_test_file(test_file)
    
    print("\nAll test files fixed!")


if __name__ == "__main__":
    main() 