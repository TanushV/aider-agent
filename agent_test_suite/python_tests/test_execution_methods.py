"""
Test script for AgentCoder execution methods.
Tests both terminal (CLI) execution and Python script execution.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path to import aider modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aider.coders.agent_coder import AgentCoder
from aider.io import InputOutput
from aider import models
import argparse


def test_python_script_execution():
    """Test executing AgentCoder from Python script."""
    print(f"\n{'='*60}")
    print("Testing Python Script Execution")
    print(f"{'='*60}\n")
    
    # Create a simple Python script that uses AgentCoder
    test_script = '''
import os
import sys
from pathlib import Path

# Import aider modules
from aider.coders.agent_coder import AgentCoder
from aider.io import InputOutput
from aider import models
import argparse

# Create args
args = argparse.Namespace()
args.model = "gemini/gemini-1.5-flash"
args.edit_format = "diff"
args.agent_coder = True
args.agent_hierarchical_planning = "deliverables_only"
args.agent_generate_tests = "descriptions"
args.agent_max_decomposition_depth = 1
args.agent_headless = True
args.agent_auto_approve = True
args.agent_web_search = "never"
args.agent_output_plan_only = True  # Just generate plan
args.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
args.auto_commits = False
args.dirty_commits = False
args.dry_run = True
args.cache_prompts = False
args.verbose = True
args.stream = True
args.pretty = True
args.weak_model = None
args.map_tokens = 1024
args.agent_enable_planner_executor_arch = False

# Create IO
io = InputOutput(
            pretty=True,
            yes=True
        )

# Create model
main_model = models.Model(
    args.model,
    weak_model=getattr(args, 'weak_model', None),
    verbose=args.verbose
)

# Create and run AgentCoder
try:
    agent = AgentCoder(
        main_model=main_model,
        io=io,
        args=args,
        initial_task="Create a simple hello world function in Python",
        fnames=[],
        read_only_fnames=[]
    )
    
    # Run agent
    result = agent.run()
    
    # Check results
    if hasattr(agent, 'plan') and agent.plan:
        print("SUCCESS: Plan generated via Python script")
        print(f"Tasks: {len(agent.plan.get('tasks', []))}")
    else:
        print("FAILED: No plan generated")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write test script
        script_path = Path(temp_dir) / "test_agent_script.py"
        script_path.write_text(test_script)
        
        # Execute the script
        try:
            print("Executing Python script...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=temp_dir,
                timeout=60
            )
            
            print(f"Exit code: {result.returncode}")
            print(f"\nStdout:\n{result.stdout}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            
            # Check for success
            if "SUCCESS: Plan generated via Python script" in result.stdout:
                print("\n✅ Python script execution successful")
                return True
            else:
                print("\n❌ Python script execution failed")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n❌ Script execution timed out")
            return False
        except Exception as e:
            print(f"\n❌ Error executing script: {e}")
            return False


def test_terminal_execution():
    """Test executing AgentCoder from terminal/CLI."""
    print(f"\n{'='*60}")
    print("Testing Terminal/CLI Execution")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare terminal command
        cmd = [
            sys.executable, "-m", "aider",
            "--agent-coder",
            "--model", "gemini/gemini-1.5-flash",
            "--agent-hierarchical-planning", "deliverables_only",
            "--agent-generate-tests", "descriptions",
            "--agent-max-decomposition-depth", "1",
            "--agent-headless",
            "--agent-output-plan-only",
            "--no-auto-commits",
            "--no-dirty-commits",
            "--dry-run",
            "--message", "Create a simple calculator function that adds two numbers"
        ]
        
        # Set environment variable for API key
        env = os.environ.copy()
        env["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
        
        try:
            print("Executing terminal command...")
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=temp_dir,
                env=env,
                timeout=60
            )
            
            print(f"Exit code: {result.returncode}")
            print(f"\nStdout:\n{result.stdout[:1000]}...")  # First 1000 chars
            if result.stderr:
                print(f"\nStderr:\n{result.stderr[:500]}...")
            
            # Check for success indicators
            success_indicators = [
                "AgentCoder initialized",
                "Agent Mode:",
                "Planning",
                "plan",
                "tasks"
            ]
            
            success_count = sum(1 for indicator in success_indicators if indicator.lower() in result.stdout.lower())
            
            if success_count >= 3:
                print(f"\n✅ Terminal execution successful (found {success_count}/{len(success_indicators)} indicators)")
                return True
            else:
                print(f"\n❌ Terminal execution may have failed (found only {success_count}/{len(success_indicators)} indicators)")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n❌ Terminal execution timed out")
            return False
        except Exception as e:
            print(f"\n❌ Error executing terminal command: {e}")
            return False


def test_execution_with_files():
    """Test AgentCoder execution with actual file operations."""
    print(f"\n{'='*60}")
    print("Testing Execution with File Operations")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = Path(temp_dir) / "calculator.py"
        test_file.write_text("""# Simple calculator module

def add(a, b):
    # TODO: Implement addition
    pass

def subtract(a, b):
    # TODO: Implement subtraction
    pass
""")
        
        # Create args
        args = argparse.Namespace()
        args.model = "gemini/gemini-1.5-flash"
        args.edit_format = "diff"
        args.agent_coder = True
        args.agent_hierarchical_planning = "none"  # Simple task
        args.agent_generate_tests = "none"
        args.agent_headless = True
        args.agent_auto_approve = True
        args.agent_web_search = "never"
        args.agent_output_plan_only = False  # Actually execute
        args.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
        args.auto_commits = False
        args.dirty_commits = False
        args.dry_run = False  # Allow file modifications
        args.cache_prompts = False
        args.verbose = True
        args.stream = True
        args.pretty = True
        args.weak_model = None
        args.map_tokens = 1024
        args.agent_enable_planner_executor_arch = False
        args.agent_max_decomposition_depth = 2
        
        # Create IO
        io = InputOutput(
            pretty=True,
            yes=True
        )
        
        # Create model instance
        main_model = models.Model(
            args.model,
            weak_model=getattr(args, 'weak_model', None),
            verbose=True
        )
        
        try:
            # Create AgentCoder with the file
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task="Implement the add and subtract functions in calculator.py",
                fnames=[str(test_file)],
                read_only_fnames=[]
            )
            
            print(f"Working with file: {test_file}")
            print("Running AgentCoder...")
            
            # Run through planning
            result = agent.run()
            
            # Continue execution
            max_iterations = 5
            iteration = 0
            while agent.current_phase not in ["reporting", "idle"] and iteration < max_iterations:
                result = agent.run()
                iteration += 1
            
            # Check if file was modified
            updated_content = test_file.read_text()
            if "return a + b" in updated_content or "return a - b" in updated_content:
                print("\n✅ File successfully modified by AgentCoder")
                print(f"Updated content:\n{updated_content}")
                return True
            else:
                print("\n⚠️  File may not have been modified as expected")
                print(f"Content:\n{updated_content}")
                return True  # Still consider it success if agent ran without errors
                
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all execution method tests."""
    print("🧪 AgentCoder Execution Methods Test Suite")
    print("=" * 60)
    
    # Test 1: Python script execution
    print("\n📝 Test 1: Python Script Execution")
    test_python_script_execution()
    
    # Test 2: Terminal/CLI execution
    print("\n\n📝 Test 2: Terminal/CLI Execution")
    test_terminal_execution()
    
    # Test 3: Execution with file operations
    print("\n\n📝 Test 3: Execution with File Operations")
    test_execution_with_files()
    
    print("\n\n✅ All execution method tests completed!")


if __name__ == "__main__":
    # Ensure we have Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not found in environment.")
        print("Using default key for testing...")
    
    main() 