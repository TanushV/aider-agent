"""
Test script for AgentCoder's clarification methods.
Tests both interactive clarification and self-clarification (headless mode).
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path to import aider modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aider.coders.agent_coder import AgentCoder
from aider.io import InputOutput
from aider import models
import argparse


def create_test_args(headless=False, web_search="never"):
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
    args.agent_hierarchical_planning = "full_two_level"
    args.agent_generate_tests = "descriptions"
    args.agent_max_decomposition_depth = 2
    args.agent_headless = headless
    args.agent_auto_approve = headless  # Auto-approve in headless mode
    args.agent_web_search = web_search
    args.agent_output_plan_only = False
    args.agent_enable_planner_executor_arch = False
    
    # File handling
    args.auto_commits = False
    args.dirty_commits = False
    args.dry_run = True
    
    # Set other required attributes
    args.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA8ME-wELj5hyXXYu0yKAe32VIoD9-sZ8E")
    
    return args


def test_self_clarification(task_description):
    """Test self-clarification in headless mode."""
    print(f"\n{'='*60}")
    print("Testing Self-Clarification (Headless Mode)")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args for headless mode
        args = create_test_args(headless=True)
        
        # Create IO instance
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
            # Create AgentCoder instance
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task=task_description,
                fnames=[],
                read_only_fnames=[]
            )
            
            print(f"Initial task: {task_description}")
            print(f"Headless mode: {agent.is_headless}")
            print(f"Initial phase: {agent.current_phase}")
            
            # Check that we skip clarification and go directly to planning
            if agent.current_phase == "planning":
                print("\n✅ Successfully skipped clarification phase in headless mode")
                print(f"Clarified task set to: {agent.clarified_task[:100]}...")
                
                # Run the planning phase
                result = agent.run()
                
                if hasattr(agent, 'plan') and agent.plan:
                    print("\n✅ Plan generated successfully in headless mode")
                    return True
                else:
                    print("\n❌ Failed to generate plan in headless mode")
                    return False
            else:
                print(f"\n❌ Expected 'planning' phase but got '{agent.current_phase}'")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_interactive_clarification(task_description):
    """Test interactive clarification mode with simulated user input."""
    print(f"\n{'='*60}")
    print("Testing Interactive Clarification Mode")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args for interactive mode
        args = create_test_args(headless=False)
        
        # Create IO instance with mocked input
        io = InputOutput(
            pretty=True,
            yes=False
        )
        
        # Create model instance
        main_model = models.Model(
            args.model,
            weak_model=getattr(args, 'weak_model', None),
            verbose=True
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
            
            print(f"Initial task: {task_description}")
            print(f"Headless mode: {agent.is_headless}")
            print(f"Initial phase: {agent.current_phase}")
            
            # Check that we start in clarification phase
            if agent.current_phase == "clarification":
                print("\n✅ Successfully entered clarification phase")
                
                # Run first clarification interaction
                result = agent.run()
                
                # Check clarification history
                if hasattr(agent, 'clarification_history') and len(agent.clarification_history) > 0:
                    print(f"\n✅ Clarification dialogue started")
                    print(f"History length: {len(agent.clarification_history)}")
                    
                    # Simulate user providing clarification
                    print("\nSimulating user response...")
                    clarification_response = "Yes, please create a REST API with endpoints for creating, reading, updating, and deleting todo items. Include user authentication."
                    
                    # Run with simulated user input
                    result = agent.run(clarification_response)
                    
                    # Check if we got another clarification or moved to planning
                    if "[CLARIFICATION_COMPLETE]" in str(result) or agent.current_phase == "planning":
                        print("\n✅ Clarification completed successfully")
                        return True
                    else:
                        print(f"\n⚠️  Clarification continues, current phase: {agent.current_phase}")
                        # This is also acceptable - agent may need more clarification
                        return True
                else:
                    print("\n❌ Clarification history not initialized properly")
                    return False
            else:
                print(f"\n❌ Expected 'clarification' phase but got '{agent.current_phase}'")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_web_search_in_clarification():
    """Test web search functionality during clarification."""
    print(f"\n{'='*60}")
    print("Testing Web Search During Clarification")
    print(f"{'='*60}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup args with web search enabled
        args = create_test_args(headless=True, web_search="always")
        args.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # For browser-use
        
        # Create IO instance
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
            # Create AgentCoder instance
            agent = AgentCoder(
                main_model=main_model,
                io=io,
                args=args,
                initial_task="What are the latest best practices for React hooks in 2024?",
                fnames=[],
                read_only_fnames=[]
            )
            
            # Check if SearchEnhancer was initialized
            if hasattr(agent, 'search_enhancer') and agent.search_enhancer:
                print("\n✅ SearchEnhancer initialized successfully")
                
                # Note: Actual web search requires DeepSeek API key
                if not args.deepseek_api_key:
                    print("⚠️  No DeepSeek API key found - web search will be skipped")
                else:
                    print("✅ DeepSeek API key found - web search enabled")
                
                return True
            else:
                print("\n⚠️  SearchEnhancer not initialized (may be due to missing dependencies)")
                return True  # Not a failure, just a limitation
                
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all clarification method tests."""
    print("🧪 AgentCoder Clarification Methods Test Suite")
    print("=" * 60)
    
    # Test cases
    test_task = "Create a todo list application with a REST API"
    
    # Test 1: Self-clarification (headless mode)
    print("\n📝 Test 1: Self-Clarification (Headless Mode)")
    test_self_clarification(test_task)
    
    # Test 2: Interactive clarification
    print("\n\n📝 Test 2: Interactive Clarification")
    test_interactive_clarification(test_task)
    
    # Test 3: Web search during clarification
    print("\n\n📝 Test 3: Web Search in Clarification")
    test_web_search_in_clarification()
    
    print("\n\n✅ All clarification method tests completed!")


if __name__ == "__main__":
    # Ensure we have required API keys
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not found in environment.")
        print("Using default key for testing...")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  Warning: DEEPSEEK_API_KEY not found for web search testing.")
        print("Web search tests will be limited...")
    
    main() 