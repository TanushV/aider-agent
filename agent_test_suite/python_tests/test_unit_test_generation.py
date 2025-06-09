"""
Test script for AgentCoder's automatic unit test generation feature.
Tests different test generation modes and verifies test creation.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import aider modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aider.coders.agent_coder import AgentCoder
from aider.io import InputOutput
from aider import models
import argparse


def create_test_args(test_generation_mode="all_code", hierarchical_mode="full_two_level"):
    """Create argparse Namespace with agent configuration."""
    args = argparse.Namespace()
    
    # Basic configuration
    args.model = "gemini/gemini-1.5-flash"
    args.edit_format = "diff"
    args.weak_model = None
    args.map_tokens = 1024
    args.cache_prompts = False
    args.verbose = True
    args.stream = True
    args.pretty = True
    
    # Agent-specific configuration
    args.agent_coder = True
    args.agent_hierarchical_planning = hierarchical_mode
    args.agent_generate_tests = test_generation_mode
    args.agent_max_decomposition_depth = 2
    args.agent_headless = True  # For automated testing
    args.agent_auto_approve = True
    args.agent_web_search = "never"  # Disable for this test
    args.agent_output_plan_only = False  # We want to see the test generation
    args.agent_enable_planner_executor_arch = False
    
    # File handling
    args.auto_commits = False
    args.dirty_commits = False
    args.dry_run = False  # We want to see actual test files generated
    
    # Set other required attributes
    args.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
    
    return args


def test_unit_test_generation(test_mode, task_description, expected_test_type):
    """Test unit test generation with specific mode."""
    print(f"\n{'='*60}")
    print(f"Testing Unit Test Generation - Mode: {test_mode}")
    print(f"{'='*60}\n")
    
    # Create test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args
        args = create_test_args(
            test_generation_mode=test_mode,
            hierarchical_mode="full_two_level" if test_mode != "none" else "none"
        )
        
        # Create IO instance
        io = InputOutput(
            pretty=True,
            yes=True  # Auto-approve
        )
        
        # Create model instance
        main_model = models.Model(
            args.model,
            weak_model=getattr(args, 'weak_model', None),
            verbose=args.verbose
        )
        
        try:
            # Create AgentCoder instance
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task=task_description,
                fnames=[],
                read_only_fnames=[]
            )
            
            # Run planning phase
            print(f"Task: {task_description}")
            print(f"Expected test type: {expected_test_type}")
            print("\nStarting AgentCoder...\n")
            
            # Run through planning
            result = agent.run()
            
            # Continue to test design phase if in appropriate mode
            if test_mode != "none" and hasattr(agent, 'current_phase'):
                while agent.current_phase not in ["test_design", "approval", "idle"]:
                    result = agent.run()
                    if agent.current_phase == "idle":
                        break
            
            # Check test generation results
            if hasattr(agent, 'tests') and agent.tests:
                print("\n✅ Tests generated successfully!")
                
                # Analyze test content
                if test_mode == "descriptions":
                    print("\n📝 Test Descriptions Generated:")
                    if 'unit_tests' in agent.tests:
                        for task_id, test_desc in agent.tests['unit_tests'].items():
                            print(f"\nTask {task_id}:")
                            print(f"  {test_desc[:200]}..." if len(test_desc) > 200 else f"  {test_desc}")
                    
                    if 'integration_tests' in agent.tests:
                        print("\n📝 Integration Test Descriptions:")
                        for test_type, test_desc in agent.tests['integration_tests'].items():
                            print(f"\n{test_type}:")
                            print(f"  {test_desc[:200]}..." if len(test_desc) > 200 else f"  {test_desc}")
                
                elif test_mode == "all_code":
                    print("\n💻 Test Code Generated:")
                    
                    # Check if actual test files were created
                    test_files_found = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if 'test' in file.lower() and file.endswith('.py'):
                                test_files_found.append(os.path.join(root, file))
                    
                    if test_files_found:
                        print(f"\n✅ Found {len(test_files_found)} test files:")
                        for test_file in test_files_found:
                            print(f"  - {os.path.relpath(test_file, temp_dir)}")
                    else:
                        print("\n⚠️  No test files found in directory")
                    
                    # Show test structure from agent's tests attribute
                    if 'unit_tests' in agent.tests:
                        print(f"\n📊 Unit tests in memory: {len(agent.tests['unit_tests'])} tasks with tests")
                    if 'integration_tests' in agent.tests:
                        print(f"📊 Integration tests in memory: {len(agent.tests['integration_tests'])} test sets")
                
                return True
                
            elif test_mode == "none":
                print("\n✅ Test mode is 'none' - no tests expected or generated")
                return True
            else:
                print("\n❌ No tests were generated")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_automatic_test_generation():
    """Test that tests are generated automatically with hierarchical planning."""
    print(f"\n{'='*60}")
    print("Testing Automatic Test Generation with Hierarchical Planning")
    print(f"{'='*60}\n")
    
    # Create test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args - hierarchical planning but no explicit test generation
        args = create_test_args(
            test_generation_mode="none",  # Not explicitly requesting tests
            hierarchical_mode="full_two_level"  # But using hierarchical mode
        )
        
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
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task="Create a simple function to calculate fibonacci numbers",
                fnames=[],
                read_only_fnames=[],
                root=temp_dir
            )
            
            # Check if test generation was auto-enabled
            print(f"Initial test generation setting: {args.agent_generate_tests}")
            print(f"Agent's test generation setting: {agent.agent_generate_tests}")
            
            if agent.agent_generate_tests != "none":
                print("\n✅ Test generation was automatically enabled!")
                print(f"   Mode changed from 'none' to '{agent.agent_generate_tests}'")
                return True
            else:
                print("\n⚠️  Test generation was not automatically enabled")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def main():
    """Run all unit test generation tests."""
    print("🧪 AgentCoder Unit Test Generation Test Suite")
    print("=" * 60)
    
    # Test 1: Different test generation modes
    test_cases = [
        ("none", "Create a function to reverse a string", "No tests"),
        ("descriptions", "Create a calculator class with add, subtract, multiply, divide methods", "Test descriptions"),
        ("all_code", "Create a simple user authentication system with login and registration", "Executable test code"),
    ]
    
    print("\n📝 Test 1: Testing Different Test Generation Modes")
    for mode, task, expected in test_cases:
        test_unit_test_generation(mode, task, expected)
    
    # Test 2: Automatic test generation with hierarchical planning
    print("\n\n📝 Test 2: Testing Automatic Test Generation")
    test_automatic_test_generation()
    
    print("\n\n✅ All unit test generation tests completed!")


if __name__ == "__main__":
    # Ensure we have Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not found in environment.")
        print("Using default key for testing...")
    
    main() 