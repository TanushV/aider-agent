#!/usr/bin/env python3
"""
Fix InputOutput parameters in all test files.
"""

import os
from pathlib import Path
import re

def fix_test_file(file_path):
    """Fix InputOutput instantiation in a test file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix InputOutput instantiation - remove stream and verbose parameters
    # Pattern to match InputOutput instantiation with parameters
    pattern = r'InputOutput\(\s*pretty=True,\s*yes=True,\s*stream=True,\s*verbose=True\s*\)'
    replacement = 'InputOutput(\n            pretty=True,\n            yes=True\n        )'
    content = re.sub(pattern, replacement, content)
    
    # Also fix any other variations
    pattern2 = r'InputOutput\(\s*pretty=True,\s*yes=False,\s*stream=True,\s*verbose=True\s*\)'
    replacement2 = 'InputOutput(\n            pretty=True,\n            yes=False\n        )'
    content = re.sub(pattern2, replacement2, content)
    
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