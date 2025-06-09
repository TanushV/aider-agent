"""
Test script for AgentCoder's hierarchical decomposition feature.
Tests different depth levels and decomposition modes.
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


def create_test_args(hierarchical_mode="full_two_level", max_depth=3, generate_tests="descriptions"):
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
    args.agent_generate_tests = generate_tests
    args.agent_max_decomposition_depth = max_depth
    args.agent_headless = True  # For automated testing
    args.agent_auto_approve = True
    args.agent_web_search = "never"  # Disable for this test
    args.agent_output_plan_only = False
    args.agent_enable_planner_executor_arch = False
    
    # File handling
    args.auto_commits = False
    args.dirty_commits = False
    args.dry_run = True  # Don't actually write files for testing
    
    # Set other required attributes
    args.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
    
    return args


def test_hierarchical_decomposition(depth_level, task_description):
    """Test hierarchical decomposition at a specific depth level."""
    print(f"\n{'='*60}")
    print(f"Testing Hierarchical Decomposition - Depth Level: {depth_level}")
    print(f"{'='*60}\n")
    
    # Create test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args
        args = create_test_args(
            hierarchical_mode="full_two_level",
            max_depth=depth_level,
            generate_tests="descriptions"
        )
        
        # Create IO instance
        io = InputOutput(
            pretty=True,
            yes=True  # Auto-approve
        )
        
        # Create model instance
        main_model = models.Model(
            args.model,
            weak_model=args.weak_model if hasattr(args, 'weak_model') else None,
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
            print("\nStarting AgentCoder planning phase...\n")
            
            # Since we're in headless mode, it should go directly to planning
            result = agent.run()
            
            # Check if plan was generated
            if hasattr(agent, 'plan') and agent.plan:
                print("\n✅ Plan successfully generated!")
                print(f"\nPlan structure:")
                print(f"- Root task: {agent.plan.get('root_task_description', 'N/A')}")
                print(f"- Number of major deliverables: {len(agent.plan.get('tasks', []))}")
                
                # Analyze decomposition depth
                def count_max_depth(task_node, current_depth=0):
                    """Recursively count the maximum depth of task decomposition."""
                    if task_node.get('is_atomic', True):
                        return current_depth
                    
                    max_sub_depth = current_depth
                    for sub_task in task_node.get('sub_tasks', []):
                        sub_depth = count_max_depth(sub_task, current_depth + 1)
                        max_sub_depth = max(max_sub_depth, sub_depth)
                    
                    return max_sub_depth
                
                # Count actual depth achieved
                actual_max_depth = 0
                for task in agent.plan.get('tasks', []):
                    task_depth = count_max_depth(task)
                    actual_max_depth = max(actual_max_depth, task_depth)
                
                print(f"\n📊 Decomposition Analysis:")
                print(f"- Requested max depth: {depth_level}")
                print(f"- Actual max depth achieved: {actual_max_depth}")
                print(f"- Effective depth used: {agent.current_task_effective_depth}")
                
                # Print task hierarchy
                print("\n📋 Task Hierarchy:")
                def print_task_hierarchy(task_node, indent=0):
                    """Print task hierarchy with indentation."""
                    prefix = "  " * indent + "- "
                    print(f"{prefix}{task_node.get('description', 'No description')[:80]}...")
                    print(f"{prefix}  ID: {task_node.get('id', 'N/A')}, Atomic: {task_node.get('is_atomic', True)}")
                    
                    for sub_task in task_node.get('sub_tasks', []):
                        print_task_hierarchy(sub_task, indent + 1)
                
                for i, task in enumerate(agent.plan.get('tasks', [])):
                    print(f"\nMajor Deliverable {i+1}:")
                    print_task_hierarchy(task)
                
                return True
            else:
                print("\n❌ Failed to generate plan")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_decomposition_modes():
    """Test different hierarchical planning modes."""
    print(f"\n{'='*60}")
    print("Testing Different Hierarchical Planning Modes")
    print(f"{'='*60}\n")
    
    modes = ["none", "deliverables_only", "full_two_level"]
    task = "Create a simple todo list web application with user authentication"
    
    for mode in modes:
        print(f"\n{'='*40}")
        print(f"Testing mode: {mode}")
        print(f"{'='*40}\n")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            args = create_test_args(
                hierarchical_mode=mode,
                max_depth=3,
                generate_tests="descriptions" if mode != "none" else "none"
            )
            
            io = InputOutput(
                pretty=True,
                yes=True
            )
            # Create model instance
            main_model = models.Model(
                args.model,
                weak_model=getattr(args, 'weak_model', None),
                verbose=args.verbose
            )
            
            try:
                agent = AgentCoder(
                    main_model=main_model,
                    io=io,
                    args=args,
                    initial_task=task,
                    fnames=[],
                    read_only_fnames=[],
                    root=temp_dir
                )
                
                result = agent.run()
                
                if hasattr(agent, 'plan') and agent.plan:
                    print(f"\n✅ Mode '{mode}' - Plan generated successfully")
                    print(f"- Number of tasks: {len(agent.plan.get('tasks', []))}")
                    
                    # Check if tasks are hierarchical
                    has_subtasks = any(
                        len(task.get('sub_tasks', [])) > 0 
                        for task in agent.plan.get('tasks', [])
                    )
                    print(f"- Has hierarchical structure: {has_subtasks}")
                else:
                    print(f"\n❌ Mode '{mode}' - Failed to generate plan")
                    
            except Exception as e:
                print(f"\n❌ Mode '{mode}' - Error: {e}")


def main():
    """Run all hierarchical decomposition tests."""
    print("🧪 AgentCoder Hierarchical Decomposition Test Suite")
    print("=" * 60)
    
    # Test 1: Different depth levels
    test_cases = [
        (1, "Create a simple calculator application"),
        (2, "Build a weather forecast web application with API integration"),
        (3, "Develop a full-featured blog platform with user authentication, posts, comments, and admin panel"),
        (4, "Create an e-commerce platform with product catalog, shopping cart, payment processing, and order management"),
    ]
    
    print("\n📝 Test 1: Testing Different Depth Levels")
    for depth, task in test_cases:
        test_hierarchical_decomposition(depth, task)
    
    # Test 2: Different decomposition modes
    print("\n\n📝 Test 2: Testing Different Decomposition Modes")
    test_decomposition_modes()
    
    print("\n\n✅ All hierarchical decomposition tests completed!")


if __name__ == "__main__":
    # Ensure we have Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not found in environment.")
        print("Using default key for testing...")
    
    main() 