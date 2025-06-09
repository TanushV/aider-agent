#!/bin/bash
# Test script for basic AgentCoder functionality from terminal

echo "🧪 Testing Basic AgentCoder from Terminal"
echo "========================================"

# Set up environment
export GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E}"

# Create temporary directory for tests
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo -e "\n📝 Test 1: Simple task with flat planning"
echo "----------------------------------------"
python -m aider \
    --agent-coder \
    --model gemini/gemini-1.5-flash \
    --agent-hierarchical-planning none \
    --agent-generate-tests none \
    --agent-headless \
    --agent-output-plan-only \
    --no-auto-commits \
    --dry-run \
    --message "Create a function to calculate the area of a circle"

echo -e "\n📝 Test 2: Hierarchical planning with test descriptions"
echo "-----------------------------------------------------"
python -m aider \
    --agent-coder \
    --model gemini/gemini-1.5-flash \
    --agent-hierarchical-planning full_two_level \
    --agent-generate-tests descriptions \
    --agent-max-decomposition-depth 2 \
    --agent-headless \
    --agent-output-plan-only \
    --no-auto-commits \
    --dry-run \
    --message "Build a simple calculator with basic operations"

echo -e "\n📝 Test 3: Full execution with file creation"
echo "-------------------------------------------"
echo "# Empty calculator module" > calculator.py

python -m aider \
    --agent-coder \
    --model gemini/gemini-1.5-flash \
    --agent-hierarchical-planning deliverables_only \
    --agent-generate-tests all_code \
    --agent-headless \
    --no-auto-commits \
    --file calculator.py \
    --message "Implement add, subtract, multiply, and divide functions in calculator.py"

# Check if file was modified
if grep -q "def add" calculator.py; then
    echo "✅ File was successfully modified"
    cat calculator.py
else
    echo "❌ File was not modified as expected"
fi

# Clean up
cd ..
rm -rf "$TEMP_DIR"

echo -e "\n✅ Terminal tests completed" 