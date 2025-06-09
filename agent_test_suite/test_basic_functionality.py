#!/usr/bin/env python3
"""
Simple test to verify basic AgentCoder functionality.
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aider.coders.agent_coder import AgentCoder
from aider.io import InputOutput
from aider import models
import argparse


def test_basic_agentcoder():
    """Test basic AgentCoder instantiation and configuration."""
    print("Testing Basic AgentCoder Functionality")
    print("=" * 50)
    
    # Set up environment
    os.environ["GEMINI_API_KEY"] = "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E"
    
    # Create args
    args = argparse.Namespace()
    args.model = "gemini/gemini-1.5-flash"
    args.edit_format = "diff"
    args.weak_model = None
    args.verbose = True
    args.cache_prompts = False
    args.map_tokens = 1024
    
    # Agent-specific settings
    args.agent_coder = True
    args.agent_hierarchical_planning = "deliverables_only"
    args.agent_generate_tests = "descriptions"
    args.agent_max_decomposition_depth = 2
    args.agent_headless = True
    args.agent_auto_approve = True
    args.agent_web_search = "never"
    args.agent_output_plan_only = True
    args.agent_enable_planner_executor_arch = False
    args.gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # File handling
    args.auto_commits = False
    args.dirty_commits = False
    args.dry_run = True
    
    print("\n1. Testing Model Creation...")
    try:
        main_model = models.Model(
            args.model,
            weak_model=None,
            verbose=args.verbose
        )
        print(f"✅ Model created: {main_model.name}")
        print(f"   Edit format: {main_model.edit_format}")
        print(f"   Max tokens: {main_model.info.get('max_input_tokens', 'N/A')}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing InputOutput Creation...")
    try:
        io = InputOutput(pretty=True, yes=True)
        print("✅ InputOutput created successfully")
    except Exception as e:
        print(f"❌ Failed to create InputOutput: {e}")
        return False
    
    print("\n3. Testing AgentCoder Creation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task="Create a simple hello world function",
                fnames=[],
                read_only_fnames=[],
                root=temp_dir
            )
            print("✅ AgentCoder created successfully")
            print(f"   Initial phase: {agent.current_phase}")
            print(f"   Headless mode: {agent.is_headless}")
            print(f"   Hierarchical planning: {agent.agent_hierarchical_planning}")
            print(f"   Test generation: {agent.agent_generate_tests}")
            
            # Test that test generation is auto-enabled
            if agent.agent_hierarchical_planning != "none" and agent.agent_generate_tests != "none":
                print("✅ Test generation auto-enabled with hierarchical planning")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create AgentCoder: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_terminal_command():
    """Test running AgentCoder from command line."""
    print("\n\nTesting Terminal Command Execution")
    print("=" * 50)
    
    import subprocess
    
    cmd = [
        sys.executable, "-m", "aider",
        "--agent-coder",
        "--model", "gemini/gemini-1.5-flash",
        "--agent-hierarchical-planning", "none",
        "--agent-headless",
        "--agent-output-plan-only",
        "--no-auto-commits",
        "--dry-run",
        "--message", "Create a function to add two numbers",
        "--exit"
    ]
    
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E"
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.abspath(os.path.join(os.getcwd(), "..")),
            timeout=30
        )
        
        print(f"\nExit code: {result.returncode}")
        
        if "AgentCoder initialized" in result.stdout:
            print("✅ AgentCoder initialized via terminal command")
            return True
        else:
            print("❌ AgentCoder may not have initialized properly")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🧪 AgentCoder Basic Functionality Tests")
    print("=" * 50)
    
    # Test 1: Basic instantiation
    test1_result = test_basic_agentcoder()
    
    # Test 2: Terminal command
    test2_result = test_terminal_command()
    
    # Summary
    print("\n\nTest Summary")
    print("=" * 50)
    print(f"Basic instantiation: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Terminal command: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        print("\n✅ All basic tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 