from aider.coders.base_coder import Coder
from aider.commands import SwitchCoder # May need to adjust import based on final location
from aider import prompts, urls, utils # Added utils
from aider.io import InputOutput
from aider.mdstream import MarkdownStream
import json # For parsing LLM plan output
import os
import re
import subprocess
from pathlib import Path
import sys
import platform
import tempfile
import traceback

import aider.models # Added for model instantiation

from aider.prompts import (
    agent_coding_system,
    agent_debugging_system,
    agent_clarification_system,
    agent_planning_system,
    agent_reporting_system,
    agent_task_review_system,
    agent_test_design_system,
    agent_integration_debugging_system,
    agent_test_command_system, # Added
    agent_decompose_deliverable_system,
    agent_generate_unit_tests_system, # Added
    agent_generate_integration_tests_for_major_deliverable_system, # Added
    agent_generate_overall_integration_tests_system, # Added
    agent_estimate_decomposition_depth_system,
    agent_recursive_decompose_task_system,
    agent_analyze_error_system,
    agent_implement_fix_plan_system,
    agent_analyze_integration_error_system,
)
from aider.search_enhancer import SearchEnhancer

# Constants for phase transitions, can be centralized if used elsewhere
MAX_CODING_RETRIES_PER_DELIVERABLE = 3
MAX_INTEGRATION_TEST_RETRIES = 2 # Increased from 1

class AgentCoder(Coder):
    coder_name = "agent" # For identification, if needed
    announce_after_switch = False # Agent will manage its own announcements

    def __init__(self, main_model, io, **kwargs):
        # Pop AgentCoder-specific arguments from kwargs
        self.initial_task = kwargs.pop("initial_task", "No task provided.")
        self.project_context_token_budget = kwargs.pop("project_context_token_budget", None)
        self.allow_project_context = kwargs.pop("allow_project_context", True)
        self.max_project_context_prompts = kwargs.pop("max_project_context_prompts", 1)
        self.allow_retries = kwargs.pop("allow_retries", True)
        self.max_retries = kwargs.pop("max_retries", 3)
        self.fail_on_error = kwargs.pop("fail_on_error", False)
        self.parent_task_id = kwargs.pop("parent_task_id", None)
        self.sub_task_id = kwargs.pop("sub_task_id", None)
        self.planner_mode = kwargs.pop("planner_mode", False)
        self.debugger_mode = kwargs.pop("debugger_mode", False)
        self.coder_mode = kwargs.pop("coder_mode", False)
        self.max_plan_depth = kwargs.pop("max_plan_depth", 3)
        self.max_search_results = kwargs.pop("max_search_results", 5)
        self.auto_apply_patches = kwargs.pop("auto_apply_patches", True)
        self.run_tests_on_change = kwargs.pop("run_tests_on_change", True)
        self.test_command = kwargs.pop("test_command", None)
        self.max_test_runs = kwargs.pop("max_test_runs", 2)
        self.debugging_context_token_budget = kwargs.pop("debugging_context_token_budget", None)
        self.max_debugging_prompts = kwargs.pop("max_debugging_prompts", 1)
        self.max_coding_prompts = kwargs.pop("max_coding_prompts", 1)
        self.execute_cmds = kwargs.pop("execute_cmds", True)
        self.confirm_cmds = kwargs.pop("confirm_cmds", False)
        # Pop 'args' before calling super, as Coder.__init__ expects it as a named param
        args_for_super = kwargs.pop('args', None)

        # from_coder is for AgentCoder internal use, not passed to Coder.__init__
        from_coder = kwargs.pop("from_coder", None)
        repo = kwargs.pop("repo", None)
        
        super().__init__(main_model, io, args_for_super, repo=repo, **kwargs)
        # self.initial_task is already set by pop
        self.current_phase = "idle" # Phases: idle, clarification, planning, test_design, approval, execution, reporting
        self.plan = None
        self.tests = None
        self.deliverables = []
        self.current_deliverable_index = 0
        self.failed_deliverables = [] # Added to track failures
        self.completed_deliverables = [] # Added to track successes
        self.agent_touched_files_rel = set() # ADDED: To track files modified by the agent
        self.overall_integration_tests_generated = False # ADDED: To track if overall integration tests have been generated
        self.integration_tests_final_status = "NOT_RUN" # Added
        self.from_coder = from_coder # Store reference to switch back
        self.clarification_history = [] # History for this phase
        self.repo_search_results = None
        self.web_search_results = None
        self.agent_test_command = None
        self.is_headless = False # Added for headless mode
        self.current_task_effective_depth = 0 # Initialize effective depth
        self.output_plan_only = False
        self.enable_planner_executor_arch = False
        self.planner_model_name = None
        self.executor_model_name = None
        self.planner_llm = self.main_model
        self.executor_llm = self.main_model
        self.search_enhancer = None

        # Store new agent config from args
        self.agent_hierarchical_planning = "none"
        self.agent_generate_tests = "none"
        self.agent_max_decomposition_depth = 3 # Default if not in args
        if self.args:
            self.agent_hierarchical_planning = getattr(self.args, 'agent_hierarchical_planning', 'none')
            self.agent_generate_tests = getattr(self.args, 'agent_generate_tests', 'none')
            self.agent_max_decomposition_depth = getattr(self.args, 'agent_max_decomposition_depth', 3)
            self.output_plan_only = getattr(self.args, 'agent_output_plan_only', False)
            self.enable_planner_executor_arch = getattr(self.args, 'agent_enable_planner_executor_arch', False)
            self.planner_model_name = getattr(self.args, 'agent_planner_model', None)
            self.executor_model_name = getattr(self.args, 'agent_executor_model', None)

            # If hierarchical planning is on and test generation is at its default 'none' (from args.py),
            # then upgrade test generation to 'descriptions' as a sensible default for hierarchical mode.
            # The user can still explicitly use --agent-generate-tests=none to disable it.
            if self.agent_hierarchical_planning != 'none' and getattr(self.args, 'agent_generate_tests') == 'none':
                # Check if 'agent_generate_tests' was explicitly provided as 'none' in command line args.
                # This is a heuristic: if 'none' is in sys.argv for this option, user likely meant it.
                # A more robust check would involve configargparse's mechanisms if available.
                user_explicitly_set_tests_to_none = False
                if "--agent-generate-tests" in sys.argv:
                    try:
                        idx = sys.argv.index("--agent-generate-tests")
                        if idx + 1 < len(sys.argv) and sys.argv[idx+1] == 'none':
                            user_explicitly_set_tests_to_none = True
                    except ValueError:
                        pass # Argument not found in simple form
                
                if not user_explicitly_set_tests_to_none:
                    self.io.tool_output(f"Agent Mode: Hierarchical planning ('{self.agent_hierarchical_planning}') is active. Defaulting test generation to 'descriptions'.")
                    self.agent_generate_tests = "descriptions"
                    if hasattr(self.args, 'agent_generate_tests'): # Keep self.args in sync
                        self.args.agent_generate_tests = "descriptions"
            else:
                 # If user specified something, respect it (it's already in self.agent_generate_tests via getattr)
                 pass

        # Inherit file context from the previous coder
        self.abs_fnames = set(kwargs.get("fnames", []))
        self.abs_read_only_fnames = set(kwargs.get("read_only_fnames", []))

        self.io.tool_output(f"AgentCoder initialized for task: {self.initial_task}")

        # Configure Planner/Executor LLMs if architecture is enabled
        if self.enable_planner_executor_arch:
            self.io.tool_output("Agent Mode: Planner/Executor architecture enabled.")
            actual_planner_model_name = self.planner_model_name if self.planner_model_name else self.main_model.name
            actual_executor_model_name = self.executor_model_name if self.executor_model_name else self.main_model.name

            self.io.tool_output(f"Agent Mode: Planner model set to: {actual_planner_model_name}")
            self.io.tool_output(f"Agent Mode: Executor model set to: {actual_executor_model_name}")

            try:
                models_factory = aider.models.Models(self.args)
                if actual_planner_model_name == self.main_model.name:
                    self.planner_llm = self.main_model
                else:
                    self.planner_llm = models_factory.get_model(actual_planner_model_name)
                    self.io.tool_output(f"Planner LLM: {self.planner_llm.name}, Edit Format: {self.planner_llm.edit_format.name}")

                if actual_executor_model_name == self.main_model.name:
                    self.executor_llm = self.main_model
                elif self.planner_llm and actual_executor_model_name == self.planner_llm.name: 
                    self.executor_llm = self.planner_llm
                else:
                    self.executor_llm = models_factory.get_model(actual_executor_model_name)
                    self.io.tool_output(f"Executor LLM: {self.executor_llm.name}, Edit Format: {self.executor_llm.edit_format.name}")

            except Exception as e:
                self.io.tool_error(f"Agent Mode: Failed to instantiate planner/executor models: {e}. Falling back to main model for all roles.")
                self.enable_planner_executor_arch = False 
                self.planner_llm = self.main_model
                self.executor_llm = self.main_model
        else:
            self.planner_llm = self.main_model
            self.executor_llm = self.main_model

        # Initialize mdstream attribute
        self.mdstream = self.io.get_assistant_mdstream()

        # Initialize SearchEnhancer if web search is enabled for the agent
        if hasattr(self.args, 'agent_web_search') and self.args.agent_web_search != "never":
            try:
                # Use planner_llm if available and P/E arch is on, otherwise main_model
                llm_for_search = self.planner_llm if self.enable_planner_executor_arch and self.planner_llm else self.main_model
                self.search_enhancer = SearchEnhancer(llm_for_search, self.io, self.args) # MODIFIED: Added self.args
                self.io.tool_output("Agent Mode: SearchEnhancer initialized for web searches.")
            except Exception as e:
                self.io.tool_error(f"Agent Mode: Failed to initialize SearchEnhancer: {e}")
                self.search_enhancer = None

        # Headless mode setup & output_plan_only implications
        if self.output_plan_only:
            self.is_headless = True # output_plan_only implies headless
            self.args.agent_auto_approve = True # and auto-approval
            self.io.tool_output("Agent Mode: Output Plan Only mode enabled (implies headless, auto-approve).")
            if hasattr(self.args, 'agent_web_search') and self.args.agent_web_search == "on_demand":
                self.io.tool_warning("Agent Mode (Output Plan Only): Web search 'on_demand' is not supported. Defaulting to 'never'.")
                self.args.agent_web_search = "never"

        elif self.args and hasattr(self.args, 'agent_headless') and self.args.agent_headless:
            self.is_headless = True
            self.io.tool_output("Agent Mode: Headless mode enabled.")
            self.args.agent_auto_approve = True # Headless implies auto-approval
            self.io.tool_output("Agent Mode (Headless): --agent-auto-approve is enabled.")
            
            if hasattr(self.args, 'agent_web_search') and self.args.agent_web_search == "on_demand":
                self.io.tool_warning("Agent Mode (Headless): Web search 'on_demand' is not supported. Defaulting to 'never'.")
                self.args.agent_web_search = "never"

        # Set initial phase based on mode
        if self.is_headless:
            self.io.tool_output("Agent Mode (Headless): Skipping clarification phase, using initial task directly.")
            self.clarified_task = self.initial_task # Use initial task directly
            self.current_phase = "planning"
        else:
            self.io.tool_output("Agent Mode: Starting task clarification phase.")
            self.current_phase = "clarification"

        # but this could be made configurable if agent could reliably output other formats.
        self.edit_format = "diff" # AgentCoder is designed to work with edit blocks

    def run(self, with_message=None):
        """
        Main entry point for the AgentCoder after it's switched to.
        This will be called by the main loop in aider/main.py.
        """
        if self.current_phase == "clarification":
            return self.run_clarification_phase(with_message)
        elif self.current_phase == "planning":
            return self.run_planning_phase()
        elif self.current_phase == "test_design":
            return self.run_test_design_phase()
        elif self.current_phase == "approval":
            return self.run_approval_phase()
        elif self.current_phase == "execution":
            return self.run_execution_phase()
        elif self.current_phase == "integration_testing":
            return self.run_integration_testing_phase()
        elif self.current_phase == "reporting":
            return self.run_reporting_phase()
        elif self.current_phase == "outputting_plan": # New phase handler
            return self.run_outputting_plan_phase()
        else:
            self.io.tool_error(f"AgentCoder in unknown phase: {self.current_phase}")
            # Potentially switch back to a default coder or end the session.
            # For now, just indicate an issue.
            return None # Or raise an exception

    # Placeholder methods for each phase - to be fleshed out
    def run_clarification_phase(self, user_input=None):
        """Handles the interactive task clarification phase."""
        system_message_content = prompts.agent_clarification_system.format(initial_task=self.initial_task)
        
        if not self.clarification_history: # First turn
            self.clarification_history.append({"role": "system", "content": system_message_content})
            # For the very first interaction, the user_input is the initial task description
            self.clarification_history.append({"role": "user", "content": self.initial_task})
            current_query_for_web_search = self.initial_task
        else:
            if user_input:
                self.clarification_history.append({"role": "user", "content": user_input})
                current_query_for_web_search = user_input
            else:
                self.io.tool_error("Agent (Clarification): Expected user input but received none after the first turn.")
                self.current_phase = "idle"
                return "Agent stopped due to missing input during clarification."

        # === Web search for clarification phase (using Browser-Use) ===
        web_search_results_str = None
        if self.search_enhancer and self.args.agent_web_search != "never":
            search_now_flag = False
            if self.args.agent_web_search == "always":
                search_now_flag = True
            elif self.args.agent_web_search == "on_demand":
                if self._confirm_action(f"Perform web search for query '{current_query_for_web_search}' during agent phase 'Clarification'?", default_response="n"):
                    search_now_flag = True
            
            if search_now_flag:
                self.io.tool_output(f"Agent (Clarification - Web Search): Using Browser-Use for: {current_query_for_web_search}")
                try:
                    web_search_results_str = self.search_enhancer.perform_browser_task(current_query_for_web_search)
                    if web_search_results_str:
                        self.io.tool_output(f"Agent (Clarification - Web Search): Found relevant information.\n{web_search_results_str[:300]}...")
                    else:
                        self.io.tool_output("Agent (Clarification - Web Search): No useful information returned from web task.")
                except Exception as e:
                    self.io.tool_error(f"Agent (Clarification - Web Search): Error during Browser-Use task: {e}")
        # === End Web search for clarification phase ===

        web_context_for_llm = ""
        if web_search_results_str: # Check the string result
            web_context_for_llm = f"\n\nRelevant information from web search:\n{web_search_results_str}"
            # Prepend to the latest user message for context, or add as a new system/user message?
            # For clarification, adding it to the system prompt might be cleaner.
            # Or, modify the current user message to include this context.
            # Let's try adding to a fresh system message for this turn for clarity.
            # This means the LLM gets: original system, history, new system (web), user question

        # Prepare messages for LLM
        # The system prompt guides the LLM to ask clarifying questions or confirm understanding.
        # The history provides the conversation so far.
        messages_for_llm = list(self.clarification_history) # Start with history (includes initial system prompt)
        if web_context_for_llm: # web_context_for_llm is now populated directly by perform_browser_task if results exist
            # Add web context as part of the user's last message, or a separate user message before LLM thinks.
            # Modifying the last user message:
            if messages_for_llm and messages_for_llm[-1]["role"] == "user":
                messages_for_llm[-1]["content"] += f"\n\nRelevant information from web search:\n{web_search_results_str}" # Use web_search_results_str directly
            else: # Should not happen if logic is correct, but as a fallback
                messages_for_llm.append({"role": "user", "content": f"Context from web search:\n{web_search_results_str}"})

        self.io.tool_output("Agent (Clarification): Thinking...")
        # Use only the main system prompt and the subsequent user/assistant turns for the actual LLM call
        # The agent_clarification_system is the primary guide here.
        # messages = [{"role": "system", "content": system_message_content}] + self.clarification_history[1:]
        # The above line might be wrong if clarification_history already contains the system message. Let's check init. It does. So, just use it.
        
        llm_response_content = self._get_llm_response(messages_for_llm, model_role="planner") 

        if not llm_response_content:
            self.io.tool_error("Agent (Clarification): Failed to get response from LLM.")
            self.current_phase = "idle"
            return "Agent stopped due to LLM error."

        self.clarification_history.append({"role": "assistant", "content": llm_response_content})

        # Check if LLM indicates completion
        # Using a simple keyword check for now, can be made more robust
        if "[CLARIFICATION_COMPLETE]" in llm_response_content or \
           "Shall I proceed to planning" in llm_response_content:

            final_understanding = llm_response_content.replace("[CLARIFICATION_COMPLETE]", "").strip()
            self.io.tool_output(f"Agent (Clarification): Understanding:\n{final_understanding}")
            self.io.tool_output("Clarification complete. Moving to planning.")
            self.current_phase = "planning"
            # Store the final clarified task (maybe the whole history or just the last summary)
            self.clarified_task = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.clarification_history])
            return self.run() # Transition to next phase immediately
        else:
            # Ask the user the LLM's question/statement
            # We don't need to show the prompt again, just the agent's response
            self.io.tool_output(f"Agent (Clarification):\\n{llm_response_content}")
            # Return None to signal the main loop to get user input
            return None

    def run_planning_phase(self):
        self.current_phase = "planning"
        self.io.tool_output("Agent Mode: Starting planning phase.")

        self.check_added_files() # Ensure file list is current. Replaced refresh_abs_fnames

        # Initial plan generation
        self.io.tool_output("Agent (Planning): Generating plan...")

        if not hasattr(self, 'clarified_task') or not self.clarified_task:
            self.io.tool_error("Agent (Planning): Clarified task is missing. Cannot generate plan.")
            self.current_phase = "idle"
            return "Agent stopped due to missing clarified task."

        # Determine effective max depth for this run based on settings
        user_configured_max_depth = self.agent_max_decomposition_depth
        llm_suggested_depth = user_configured_max_depth # Default to user-set max if LLM fails

        if self.agent_hierarchical_planning == 'none':
            self.io.tool_output("Agent (Planning): Hierarchical planning is 'none'. Effective decomposition depth will be 0.")
            self.current_task_effective_depth = 0
        else:
            # For 'deliverables_only' or 'full_two_level', we consult the LLM
            self.io.tool_output("Agent (Planning): Estimating task decomposition depth...")
            estimated_depth_prompt = prompts.agent_estimate_decomposition_depth_system.format(
                user_task_description=self.clarified_task
            )
            messages_depth_estimation = [{"role": "system", "content": estimated_depth_prompt}]
            llm_response_depth_estimation = self._get_llm_response(messages_depth_estimation, model_role="planner")
            
            if llm_response_depth_estimation:
                try:
                    parsed_depth = int(llm_response_depth_estimation.strip())
                    llm_suggested_depth = parsed_depth
                    self.io.tool_output(f"Agent (Planning): LLM initially suggested task decomposition depth: {llm_suggested_depth}")
                except ValueError:
                    self.io.tool_warning(f"Agent (Planning): Could not parse estimated depth from LLM: '{llm_response_depth_estimation}'. Using user max: {user_configured_max_depth}")
                    llm_suggested_depth = user_configured_max_depth # Fallback to user max
            else:
                self.io.tool_warning("Agent (Planning): Failed to get estimated depth from LLM. Using user max.")
                llm_suggested_depth = user_configured_max_depth # Fallback to user max

            # Now, apply constraints from agent_hierarchical_planning and agent_max_decomposition_depth
            if self.agent_hierarchical_planning == 'deliverables_only':
                # Max depth is 1 (task -> sub-tasks, sub-tasks are atomic)
                self.current_task_effective_depth = min(llm_suggested_depth, 1, user_configured_max_depth)
                self.io.tool_output(f"Agent (Planning): Mode 'deliverables_only'. Effective depth set to {self.current_task_effective_depth} (min of LLM: {llm_suggested_depth}, mode_limit: 1, user_max: {user_configured_max_depth}).")
            elif self.agent_hierarchical_planning == 'full_two_level': # Or any other mode that implies full hierarchy based on depth
                # Depth is LLM suggestion, capped by user_configured_max_depth
                self.current_task_effective_depth = min(llm_suggested_depth, user_configured_max_depth)
                self.io.tool_output(f"Agent (Planning): Mode 'full_two_level/dynamic'. Effective depth set to {self.current_task_effective_depth} (min of LLM: {llm_suggested_depth}, user_max: {user_configured_max_depth}).")
            else: # Should not happen given arg choices, but as a fallback
                self.current_task_effective_depth = min(llm_suggested_depth, user_configured_max_depth)

        # Ensure current_task_effective_depth is non-negative
        self.current_task_effective_depth = max(0, self.current_task_effective_depth)
        self.io.tool_output(f"Agent (Planning): Final effective decomposition depth for this task: {self.current_task_effective_depth}")

        # === Web search for planning phase (using Browser-Use) ===
        web_search_results_str_planning = None
        if self.search_enhancer and self.args.agent_web_search != "never":
            search_now_flag = False
            if self.args.agent_web_search == "always":
                search_now_flag = True
            elif self.args.agent_web_search == "on_demand":
                if self._confirm_action(f"Perform web search for query '{self.clarified_task}' during agent phase 'Planning'?", default_response="n"):
                    search_now_flag = True
            
            if search_now_flag:
                self.io.tool_output(f"Agent (Planning - Web Search): Using Browser-Use for: {self.clarified_task[:100]}...")
                try:
                    web_search_results_str_planning = self.search_enhancer.perform_browser_task(self.clarified_task)
                    if web_search_results_str_planning:
                         self.io.tool_output(f"Agent (Planning - Web Search): Found relevant information.\n{web_search_results_str_planning[:300]}...")
                    else:
                        self.io.tool_output("Agent (Planning - Web Search): No useful information returned from web task.")
                except Exception as e:
                    self.io.tool_error(f"Agent (Planning - Web Search): Error during Browser-Use task: {e}")
        # === End Web search for planning phase ===
        
        web_context_for_llm = ""
        if web_search_results_str_planning: # Use the planning-specific results
            web_context_for_llm = f"\n\nRelevant information from web search to consider for planning:\n{web_search_results_str_planning}"

        # The task used for the highest level of planning (recursive decomposition starts here)
        # We no longer use task_for_planning_prompt directly with agent_planning_system for major deliverables
        # Instead, clarified_task (potentially with web_context) is the input to our recursive decomposition
        initial_task_for_decomposition = self.clarified_task
        if web_context_for_llm: # Append web context if available
            initial_task_for_decomposition += web_context_for_llm
        
        self.io.tool_output(f"Agent (Planning): Starting recursive decomposition for task (max_depth={self.current_task_effective_depth}): {initial_task_for_decomposition[:100]}...")

        # === New Recursive Decomposition ===
        # The root_task_node will contain the full hierarchy.
        # IDs will be like: "task_d0", "task_d0_st1_d1", "task_d0_st1_d1_st1_d2", etc.
        # The parent_id_prefix for the initial call is simply "task"
        root_task_node = self._decompose_task_recursively(
            task_description=initial_task_for_decomposition,
            current_depth=0,
            max_depth=self.current_task_effective_depth, # This is the LLM estimated & user-capped depth
            parent_id_prefix="task" 
        )

        # Initialize the plan structure
        self.plan = {
            "root_task_description": root_task_node["description"], # Store original task desc
            "tasks": [], # These will be the old "major_deliverables"
            "overall_integration_tests": []
        }

        if root_task_node["is_atomic"]:
            # If the entire task is atomic, it's the only "major deliverable"
            self.io.tool_output("Agent (Planning): Entire task is considered atomic.")
            # We still put it in the 'tasks' list for consistency with downstream phases
            root_task_node["id"] = "md1" # Simplify ID for this single top-level task
            self.plan["tasks"] = [root_task_node]
            # Keep a flat list for older parts of the agent that might expect self.deliverables
            self.deliverables = [root_task_node["description"]]
        else:
            # If the root task was decomposed, its direct sub-tasks become the "major_deliverables"
            self.io.tool_output(f"Agent (Planning): Task decomposed into {len(root_task_node.get('sub_tasks',[]))} primary sub-tasks (major deliverables).")
            
            processed_sub_tasks = []
            for i, sub_task in enumerate(root_task_node.get("sub_tasks", [])):
                # Re-ID these top-level sub-tasks to follow "mdX" convention for compatibility
                # The sub_task itself contains its original hierarchical ID (e.g., "task_d0_st1_d1")
                # and any further nested sub_tasks.
                sub_task["original_hierarchical_id"] = sub_task["id"] # Preserve the detailed ID
                sub_task["id"] = f"md{i+1}" # Assign a simpler "major deliverable" style ID
                processed_sub_tasks.append(sub_task)
            
            self.plan["tasks"] = processed_sub_tasks
            # Keep a flat list for older parts of the agent that might expect self.deliverables
            self.deliverables = [task["description"] for task in self.plan["tasks"]]

        # The old decomposition logic based on `agent_hierarchical_planning == 'full_two_level'`
        # is now replaced by the above recursive decomposition.
        # We no longer need the section:
        # "# === Step 2: Decompose Major Deliverables (Level 2) if full_two_level ==="
        # That logic is now implicitly handled by _decompose_task_recursively if depth > 1.

        self.io.tool_output("Generated Plan (Hierarchical):")
        try:
            # Attempt to pretty-print the plan for readability
            # If self.plan becomes very large/deep, this might need adjustment or summarization
            plan_json = json.dumps(self.plan, indent=2, default=str) # Use default=str for any non-serializable items
            if len(plan_json) > 2000: # Heuristic to avoid flooding output
                 self.io.tool_output("Plan is too large to display fully. Showing summary.")
                 summary_plan = {
                     "root_task_description": self.plan.get("root_task_description"),
                     "num_major_tasks": len(self.plan.get("tasks", [])),
                     "overall_integration_tests_count": len(self.plan.get("overall_integration_tests", []))
                 }
                 self.io.tool_output(json.dumps(summary_plan, indent=2))
            else:
                self.io.tool_output(plan_json)
        except TypeError as e:
            self.io.tool_error(f"Could not serialize plan to JSON for display: {e}")
            self.io.tool_output(str(self.plan)) # Fallback to string representation


        # Initialize overall tests structure if not already present (it is, from above)
        # if "overall_integration_tests" not in self.plan:
        # self.plan["overall_integration_tests"] = []

        # self.io.tool_output("Generated Plan (Hierarchical):")
        # self.io.tool_output(json.dumps(self.plan, indent=2))
        
        # Transition to test design
        self.current_phase = "test_design"
        return self.run()

    def run_test_design_phase(self):
        self.io.tool_output("Agent (Test Design): Designing tests...")
        self.agent_test_command = None # Reset any previous command

        if not self.plan or "tasks" not in self.plan or not self.plan["tasks"]:
            self.io.tool_error("Agent (Test Design): Plan with tasks is missing. Cannot design tests.")
            self.current_phase = "idle" # Or approval to report no tests?
            return "Agent stopped due to missing plan for test design."

        if self.agent_generate_tests == "none":
            self.io.tool_output("Agent (Test Design): Test generation is set to 'none'. Skipping test design.")
            self.plan["overall_integration_tests"] = [] # Ensure key exists
            for task in self.plan["tasks"]:
                task["unit_tests"] = []
                task["integration_tests"] = []
                for sub_task in task.get("sub_tasks", []):
                    sub_task["unit_tests"] = []
            self.current_phase = "approval"
            return self.run()

        output_format_instruction = ""
        if self.agent_generate_tests == "descriptions":
            output_format_instruction = "Output *only* as a JSON list of strings, where each string describes a test case (e.g., inputs, expected outputs, conditions). Do not include any other text, explanations, or markdown formatting."
        elif self.agent_generate_tests == "all_code":
            output_format_instruction = "Output *only* as a JSON list of strings, where each string is a self-contained, runnable code snippet for a test case. The code should be for the appropriate testing framework if discernible, or generic if not. Do not include any other text, explanations, or markdown formatting outside the JSON list itself."

        # === Generate Tests based on Hierarchical Plan ===
        for task in self.plan["tasks"]:
            task_description = task["description"]
            task["unit_tests"] = [] # Initialize for the task
            task["integration_tests"] = [] # Initialize for the task (if it becomes a parent)

            # Determine if the task should be treated as non-atomic with sub-tasks for test generation
            is_decomposable_parent = not task.get("is_atomic", True) and task.get("sub_tasks")

            if is_decomposable_parent:
                self.io.tool_output(f"Agent (Test Design): Task '{task_description}' is non-atomic with sub-tasks. Generating tests for sub-tasks and integration for parent.")
                # 1. Unit tests for each sub-task
                for sub_task in task.get("sub_tasks", []):
                    sub_task_description = sub_task["description"]
                    sub_task["unit_tests"] = [] # Initialize for sub-task
                    self.io.tool_output(f"Agent (Test Design): Generating unit tests for sub-task '{sub_task_description}' of task '{task_description}'")
                    system_prompt_sub_task_units = prompts.agent_generate_unit_tests_system.format(
                        task_description=sub_task_description,
                        output_format_instruction=output_format_instruction
                    )
                    messages_sub_task_units = [{"role": "system", "content": system_prompt_sub_task_units}]
                    sub_task_test_results = self._get_llm_response(messages_sub_task_units, expecting_json=True, model_role="planner")
                    if sub_task_test_results and isinstance(sub_task_test_results, list):
                        sub_task["unit_tests"] = sub_task_test_results
                    else:
                        self.io.tool_warning(f"Failed to generate/parse unit tests for sub-task: {sub_task_description}. Response: {sub_task_test_results}")
                
                # 2. Integration tests for the parent task itself
                self.io.tool_output(f"Agent (Test Design): Generating integration tests for non-atomic task: {task_description}")
                sub_task_descs_list_str = "\n".join([f"- {st['description']}" for st in task.get("sub_tasks", [])])
                system_prompt_parent_integration = prompts.agent_generate_integration_tests_for_major_deliverable_system.format(
                    major_deliverable_description=task_description, # Keep var name for prompt compatibility
                    atomic_sub_task_descriptions_list=sub_task_descs_list_str,
                    output_format_instruction=output_format_instruction
                )
                messages_parent_integration = [{"role": "system", "content": system_prompt_parent_integration}]
                parent_integration_test_results = self._get_llm_response(messages_parent_integration, expecting_json=True, model_role="planner")
                if parent_integration_test_results and isinstance(parent_integration_test_results, list):
                    task["integration_tests"] = parent_integration_test_results
                else:
                    self.io.tool_warning(f"Failed to generate/parse integration tests for task: {task_description}. Response: {parent_integration_test_results}")
            
            else: # Task is atomic (or treated as such)
                self.io.tool_output(f"Agent (Test Design): Generating unit tests for atomic task: {task_description}")
                system_prompt_atomic_units = prompts.agent_generate_unit_tests_system.format(
                    task_description=task_description,
                    output_format_instruction=output_format_instruction
                )
                messages_atomic_units = [{"role": "system", "content": system_prompt_atomic_units}]
                atomic_test_results = self._get_llm_response(messages_atomic_units, expecting_json=True, model_role="planner")
                if atomic_test_results and isinstance(atomic_test_results, list):
                    task["unit_tests"] = atomic_test_results
                else:
                    self.io.tool_warning(f"Failed to generate/parse unit tests for atomic task: {task_description}. Response: {atomic_test_results}")

        # === Generate Overall Integration Tests for the entire request ===
        self.io.tool_output("Agent (Test Design): Generating overall integration tests for the entire request...")
        task_descs_list_str = "\n".join([f"- {task['description']}" for task in self.plan["tasks"]])
        initial_task_desc_for_prompt = self.clarified_task if self.clarified_task else self.initial_task
        
        system_prompt_overall = prompts.agent_generate_overall_integration_tests_system.format(
            initial_task_description=initial_task_desc_for_prompt,
            major_deliverables_list_description=task_descs_list_str,
            output_format_instruction=output_format_instruction
        )
        messages_overall = [{"role": "system", "content": system_prompt_overall}]
        overall_test_results = self._get_llm_response(messages_overall, expecting_json=True, model_role="planner")
        
        if "overall_integration_tests" not in self.plan:
             self.plan["overall_integration_tests"] = [] # Ensure it exists

        if overall_test_results and isinstance(overall_test_results, list):
            self.plan["overall_integration_tests"] = overall_test_results
        else:
            self.io.tool_warning(f"Failed to generate/parse overall integration tests. Response: {overall_test_results}")
            self.plan["overall_integration_tests"] = [] # Ensure it's an empty list on failure

        # === Get suggested test command from LLM (existing logic) ===
        self.io.tool_output("Agent (Test Design): Suggesting test execution command...")
        # Create a flat list of all task descriptions for the test command prompt context
        all_task_descs_for_cmd_prompt = []
        for task in self.plan["tasks"]:
            # If a task is atomic or has no sub_tasks, its description is added.
            # If it is a parent with sub_tasks, the sub_tasks' descriptions are added instead.
            if task.get("is_atomic", True) or not task.get("sub_tasks"):
                all_task_descs_for_cmd_prompt.append(task["description"])
            else: # It's a parent task, add its sub-tasks' descriptions
                for sub_task in task.get("sub_tasks", []):
                    all_task_descs_for_cmd_prompt.append(sub_task["description"])
        
        plan_overview_for_cmd = json.dumps(all_task_descs_for_cmd_prompt, indent=2)
        self.check_added_files()
        file_context_for_tests = self.get_context_for_prompts(show_abs_paths=True)

        test_command_prompt_messages = [
            {
                "role": "system",
                "content": prompts.agent_test_command_system.format(
                    plan_overview=plan_overview_for_cmd,
                    file_context_for_tests=file_context_for_tests
                )
            }
        ]
        suggested_command = self._get_llm_response(test_command_prompt_messages, model_role="planner")

        if suggested_command and isinstance(suggested_command, str):
            self.agent_test_command = suggested_command.strip()
            self.io.tool_output(f"Agent (Test Design): Suggested test command: {self.agent_test_command}")
        else:
            self.io.tool_error(f"Agent (Test Design): Failed to get a valid suggested test command from LLM. Response: {suggested_command}. Will use default if configured.")

        self.io.tool_output("Test design complete. Full test plan structure:")
        # Use a more controlled way to display the plan, respecting potential size
        try:
            plan_json_for_display = json.dumps(self.plan, indent=2, default=str)
            if len(plan_json_for_display) > 2000:
                self.io.tool_output("(Plan is large, showing summary in logs. Full plan will be in final output if --agent-output-plan-only is used)")
                summary_plan_display = {
                     "root_task_description": self.plan.get("root_task_description"),
                     "num_major_tasks": len(self.plan.get("tasks", [])),
                     "overall_integration_tests_count": len(self.plan.get("overall_integration_tests", [])),
                     "suggested_test_command": self.agent_test_command
                 }
                self.io.tool_output(json.dumps(summary_plan_display, indent=2))
            else:
                self.io.tool_output(plan_json_for_display)
        except TypeError as e:
            self.io.tool_error(f"Could not serialize plan to JSON for display during test design: {e}")
            self.io.tool_output(str(self.plan)) # Fallback

        if self.output_plan_only:
            self.current_phase = "outputting_plan"
        else:
            self.current_phase = "approval"
        return self.run()

    def run_approval_phase(self):
        """Handles the approval phase for plan and tests."""
        self.io.tool_output("Agent (Approval): Seeking approval for plan and tests...")

        if not self.plan or "tasks" not in self.plan: # Simplified check for plan existence
            self.io.tool_error("Agent (Approval): Plan missing or malformed. Cannot seek approval.")
            self.current_phase = "idle"
            return "Agent stopped due to missing/malformed plan for approval."

        # Display plan and tests derived from the plan object
        self.io.tool_output("Proposed Plan Details:")
        # Display overall task info
        self.io.tool_output(f"  Root Task: {self.plan.get('root_task_description', 'N/A')[:200]}...")
        self.io.tool_output(f"  Number of Major Deliverables: {len(self.plan.get('tasks', []))}")
        
        for i, task in enumerate(self.plan.get("tasks", [])):
            self.io.tool_output(f"\n  Deliverable {i+1}: {task.get('description', 'N/A')[:100]}...")
            if task.get("unit_tests"):
                self.io.tool_output(f"    Unit Tests ({len(task['unit_tests'])}): {json.dumps(task['unit_tests'][:2], indent=2)}...")
            else:
                self.io.tool_output("    Unit Tests: None defined.")
            if task.get("integration_tests"):
                 self.io.tool_output(f"    Integration Tests ({len(task['integration_tests'])}): {json.dumps(task['integration_tests'][:2], indent=2)}...")
            # Display sub-task tests if present
            if task.get("sub_tasks"):
                for j, sub_task in enumerate(task.get("sub_tasks",[])):
                    if sub_task.get("unit_tests"):
                        self.io.tool_output(f"      Sub-Task '{sub_task.get('description','N/A')[:50]}...' Unit Tests ({len(sub_task['unit_tests'])}): {json.dumps(sub_task['unit_tests'][:1], indent=2)}...")
        
        overall_integration_tests = self.plan.get("overall_integration_tests", [])
        if overall_integration_tests:
            self.io.tool_output(f"\n  Overall Integration Tests ({len(overall_integration_tests)}): {json.dumps(overall_integration_tests[:2], indent=2)}...")
        else:
            self.io.tool_output("\n  Overall Integration Tests: None defined.")

        if self.agent_test_command:
            self.io.tool_output(f"\nSuggested Test Command: {self.agent_test_command}")
        else:
            self.io.tool_output("\nSuggested Test Command: Not specified.")

        # Check for auto-approval
        if self.args and hasattr(self.args, 'agent_auto_approve') and self.args.agent_auto_approve:
            self.io.tool_output("Agent (Approval): Auto-approving plan and tests based on --agent-auto-approve flag.")
            self.current_phase = "execution"
            return self.run() # Transition to next phase immediately

        prompt = "Proceed with the plan? (Y/n/d<details>/h<help>) "
        if self.args and getattr(self.args, "agent_auto_approve", False):
            self.io.tool_output(f"Agent (Approval): Auto-approving: {prompt} -> Yes")
            user_response = "y"
        else:
            user_response = self.io.prompt_ask(prompt, default="y").strip().lower()

        # ==> ADD DEBUGGING HERE <==
        self.io.tool_output(f"Agent (Approval): DEBUG - user_response RAW: '{user_response}'")
        self.io.tool_output(f"Agent (Approval): DEBUG - user_response TYPE: {type(user_response)}")
        self.io.tool_output(f"Agent (Approval): DEBUG - condition (user_response == 'y'): {user_response == 'y'}")
        self.io.tool_output(f"Agent (Approval): DEBUG - condition (not user_response): {not user_response}")
        self.io.tool_output(f"Agent (Approval): DEBUG - full condition ((user_response == 'y') or (not user_response)): {(user_response == 'y') or (not user_response)}")

        if user_response == "y" or not user_response: # Empty input defaults to yes
            self.io.tool_output("Agent (Approval): Plan approved by user (or auto-approved).")
            self.current_phase = "execution"
            return self.run()
        
        elif user_response == "n":
            self.io.tool_output("Agent (Approval): Plan and tests rejected. What would you like to do? (e.g., /agent restart, /edit_plan, /quit)")
            self.current_phase = "idle"
            return "Plan rejected by user. Agent awaiting instructions."
        
        elif user_response == "d":
            self.io.tool_output("Agent (Approval): Modification requested. Please provide new instructions or /edit the plan/tests directly. The agent will re-evaluate after your changes.")
            self.current_phase = "idle"
            return "Modification requested. Agent awaiting instructions."
        
        elif user_response == "h":
            self.io.tool_output("Agent (Approval): Help requested. Please provide more details about what you need help with.")
            self.current_phase = "idle"
            return "Help requested. Agent awaiting instructions."
        
        else:
            self.io.tool_output("Agent (Approval): Invalid response. Please answer 'Y', 'n', 'd', or 'h'. Agent awaiting valid response or new instructions.")
            self.io.tool_output("Agent (Approval): DEBUG - Returning None from approval phase due to invalid user input.") # Forcing a change
            self.current_phase = "approval"
            return None

    def run_execution_phase(self):
        # Cache buster comment for agent_coder
        self.check_added_files() # Ensure file list is current
        self.io.tool_output("Agent (Execution): Starting execution of hierarchical plan...")

        if not self.plan or "tasks" not in self.plan or not self.plan["tasks"]:
            self.io.tool_error("Agent (Execution): No tasks in plan to execute.")
            self.current_phase = "reporting" 
            return self.run()

        self.completed_deliverables = [] # Tracks IDs of completed tasks
        self.failed_deliverables = []    # Tracks IDs of failed tasks
        
        original_num_tasks = len(self.plan["tasks"])

        for task_idx, task in enumerate(self.plan["tasks"]):
            task_id = task["id"]
            task_description = task["description"]
            self.io.tool_output(f"\nAgent (Execution): Starting task {task_idx + 1}/{original_num_tasks}: '{task_description}' (ID: {task_id})")
            
            task_completed_successfully = False

            # Determine if the task should be treated as non-atomic with sub-tasks for execution
            is_decomposable_parent_with_subtasks = not task.get("is_atomic", True) and task.get("sub_tasks")

            if is_decomposable_parent_with_subtasks:
                # === Non-Atomic task: Process Sub-Tasks ===
                self.io.tool_output(f"Agent (Execution): Task '{task_description}' is non-atomic with sub-tasks. Processing its sub-tasks.")
                all_sub_tasks_completed = True
                num_sub_tasks = len(task["sub_tasks"])

                for sub_task_idx, sub_task_node in enumerate(task["sub_tasks"]):
                    # Each sub_task_node here is a full task dictionary from the recursive decomposition
                    sub_task_id = sub_task_node["id"]
                    sub_task_desc = sub_task_node["description"]
                    self.io.tool_output(f"Agent (Execution):  Executing Sub-Task {sub_task_idx + 1}/{num_sub_tasks}: '{sub_task_desc}' (ID: {sub_task_id}) for parent '{task_description}'")
                    
                    # Recursively execute this sub-task if it also has sub-tasks, 
                    # or execute it as atomic if it doesn't.
                    # This requires a new helper or modification to _execute_atomic_task to handle hierarchies.
                    # For now, let's assume _execute_atomic_task can handle a task_node that might have sub_tasks,
                    # but it will only act on it if it's marked atomic. If sub_task_node itself is a parent,
                    # this current loop structure needs to be recursive for execution too.

                    # Simplification for now: The current _execute_atomic_task expects a flat task. 
                    # The sub_tasks here are the *actual* atomic units if the plan was fully decomposed to its max depth.
                    # If a sub_task was NOT decomposed further (because max_depth was met at its level, or it was deemed atomic),
                    # then sub_task_node["is_atomic"] will be true.
                    # If a sub_task *could* have been decomposed further but wasn't due to planning depth limits, it might be complex.

                    # The _execute_atomic_task is designed for truly atomic pieces of work.
                    # If sub_task_node itself is a parent (not atomic and has sub_tasks), we need recursive execution.
                    # Let's create _execute_hierarchical_task(task_node) helper.

                    # For this iteration, we will assume tasks in task["sub_tasks"] are meant to be executed atomically.
                    # If they themselves have sub-tasks (meaning current_task_effective_depth allowed deeper plans),
                    # the _execute_atomic_task will need to be smart or we need a recursive execution wrapper here.
                    # Current `_execute_atomic_task` will take the `sub_task_node` and only use its description and tests.

                    sub_task_to_execute = {
                        "id": sub_task_id,
                        "description": sub_task_desc,
                        "unit_tests": sub_task_node.get("unit_tests", []),
                        "type": f"Sub-Task of {task_id}" # More descriptive type
                    }
                    # We pass the parent description for context to _execute_atomic_task
                    sub_task_success = self._execute_task_recursively(sub_task_to_execute, parent_description=task_description)
                    
                    if not sub_task_success:
                        all_sub_tasks_completed = False
                        self.io.tool_error(f"Agent (Execution): Sub-Task '{sub_task_desc}' failed. Aborting further sub-tasks for parent task '{task_description}'.")
                        break # Stop processing further sub-tasks for this parent task
                
                if all_sub_tasks_completed:
                    self.io.tool_output(f"Agent (Execution): All sub-tasks for parent task '{task_description}' completed.")
                    # Now, run integration tests for this parent task if they exist
                    parent_integration_tests = task.get("integration_tests", [])
                    if parent_integration_tests:
                        self.io.tool_output(f"Agent (Execution): Running integration tests for parent task '{task_description}'. Test descriptions/code: {json.dumps(parent_integration_tests)}")
                        # _run_tests uses the general test command. The specific tests are for LLM guidance.
                        test_output, test_success = self._run_tests(is_integration_test=True)
                        if test_success:
                            self.io.tool_output(f"Agent (Execution): Integration tests for parent task '{task_description}' PASSED.")
                            task_completed_successfully = True
                        else:
                            self.io.tool_error(f"Agent (Execution): Integration tests for parent task '{task_description}' FAILED. Output:\n{test_output}")
                            # Attempt to debug parent task integration failures

                            # Begin debugging loop for parent task integration failure
                            max_parent_integration_retries = MAX_INTEGRATION_TEST_RETRIES # Use the same constant
                            parent_integration_attempt = 0
                            parent_integration_fixed = False

                            current_test_output = test_output # Store the initial failing output

                            while parent_integration_attempt < max_parent_integration_retries and not parent_integration_fixed:
                                parent_integration_attempt += 1
                                self.io.tool_output(f"Agent (Execution - Parent Integration Debug): Attempt {parent_integration_attempt}/{max_parent_integration_retries} to fix integration tests for parent task '{task_description}'...")

                                # Prepare context for debugging
                                self.check_added_files()
                                code_context_for_parent_debug = self.get_context_for_prompts(show_abs_paths=True)
                                plan_overview_for_parent_debug = json.dumps(self.plan, indent=2) # Overview of the whole plan
                                failed_test_desc_for_parent = f"Integration tests for parent task '{task_description}' (part of overall plan) failed."
                                
                                parent_it_web_search_query = f"Fix integration test error for parent task {task_description}: {current_test_output[:200]}"
                                # === Web search for parent IT debug ===
                                parent_it_web_results_str = None
                                if self.search_enhancer and self.args.agent_web_search != "never":
                                    search_now_flag = False
                                    # For debugging, let's assume 'always' or pre-approved 'on_demand' for now to simplify, 
                                    # or make it 'always' if in a debugging loop.
                                    # Let's lean towards 'always' during debugging if search is generally enabled.
                                    if self.args.agent_web_search == "always" or self.args.agent_web_search == "on_demand": # Simplified for debug
                                        search_now_flag = True # For on_demand, assume user would want it in a fix loop
                                    
                                    if search_now_flag:
                                        self.io.tool_output(f"Agent (Execution - Parent IT Debug - Web Search): Using Browser-Use for: {parent_it_web_search_query}")
                                        try:
                                            parent_it_web_results_str = self.search_enhancer.perform_browser_task(parent_it_web_search_query)
                                            if parent_it_web_results_str:
                                                self.io.tool_output(f"Agent (Execution - Parent IT Debug - Web Search): Found relevant information.\n{parent_it_web_results_str[:300]}...")
                                            else: self.io.tool_output("Agent (Execution - Parent IT Debug - Web Search): No useful extracts.")
                                        except Exception as e:
                                            self.io.tool_error(f"Agent (Execution - Parent IT Debug - Web Search): Error: {e}")
                                # === End Web search for parent IT debug ===
                                parent_it_web_context_str = parent_it_web_results_str if parent_it_web_results_str else "No web search performed or no results found."
                                
                                llm_fix_response_parent = None

                                if self.enable_planner_executor_arch:
                                    self.io.tool_output(f"Agent (Execution - Parent IT Debug): Using Planner/Executor flow for parent task '{task_description}'.")
                                    analysis_prompt_content = prompts.agent_analyze_integration_error_system.format(
                                        plan_overview=plan_overview_for_parent_debug,
                                        failed_test_description=failed_test_desc_for_parent,
                                        test_output_and_errors=current_test_output,
                                        code_context=code_context_for_parent_debug,
                                        web_search_context=parent_it_web_context_str
                                    )
                                    analysis_messages = [{"role": "system", "content": analysis_prompt_content}]
                                    planner_analysis_json = self._get_llm_response(analysis_messages, expecting_json=True, model_role="planner")

                                    if planner_analysis_json and isinstance(planner_analysis_json, dict) and all(k in planner_analysis_json for k in ["error_analysis", "fix_plan", "target_files_and_lines_suggestion"]):
                                        error_analysis = planner_analysis_json["error_analysis"]
                                        fix_plan = planner_analysis_json["fix_plan"]
                                        suggestion = planner_analysis_json["target_files_and_lines_suggestion"]
                                        self.io.tool_output(f"Agent (Planner Analysis - Parent IT): Error: {error_analysis}. Plan: {fix_plan}. Suggestion: {suggestion}")

                                        general_task_desc_for_fix = f"Fix integration test failures for parent task '{task_description}' which is part of the overall task: {self.clarified_task if self.clarified_task else self.initial_task}"
                                        implement_fix_prompt_content = prompts.agent_implement_fix_plan_system.format(
                                            task_description=general_task_desc_for_fix,
                                            failed_code_attempt="Multiple files may have changed, see context.",
                                            test_output_and_errors=current_test_output,
                                            error_analysis_from_planner=error_analysis,
                                            fix_plan_from_planner=fix_plan,
                                            target_files_and_lines_suggestion_from_planner=suggestion,
                                            file_context=code_context_for_parent_debug
                                        )
                                        implement_fix_messages = [{"role": "system", "content": implement_fix_prompt_content}]
                                        llm_fix_response_parent = self._get_llm_response(implement_fix_messages, model_role="executor")
                                    else:
                                        self.io.tool_warning("Agent (Planner Analysis - Parent IT): Failed to get valid analysis from Planner LLM.")
                                else: # Legacy debugging flow
                                    self.io.tool_output(f"Agent (Execution - Parent IT Debug): Using legacy debugging flow for parent task '{task_description}'.")
                                    legacy_debug_prompt = prompts.agent_integration_debugging_system.format(
                                        plan_overview=plan_overview_for_parent_debug,
                                        failed_test_description=failed_test_desc_for_parent,
                                        test_output_and_errors=current_test_output,
                                        web_search_context=parent_it_web_context_str,
                                        code_context=code_context_for_parent_debug,
                                    )
                                    legacy_debug_messages = [{"role": "system", "content": legacy_debug_prompt}]
                                    llm_fix_response_parent = self._get_llm_response(legacy_debug_messages, model_role="planner") # Planner for legacy debug

                                if not llm_fix_response_parent:
                                    self.io.tool_error("Agent (Execution - Parent IT Debug): LLM failed to provide a fix. Retrying tests as is (if attempts left).")
                                    # Re-run tests to see if issue was transient, or if we should break
                                    current_test_output, test_success_after_failed_fix_attempt = self._run_tests(is_integration_test=True)
                                    if test_success_after_failed_fix_attempt:
                                        parent_integration_fixed = True
                                    continue # To next attempt or break if max retries

                                applied_fix_info = self._parse_and_apply_edits(llm_fix_response_parent)
                                if not applied_fix_info["applied_paths"] and not applied_fix_info["created_paths"]:
                                    self.io.tool_error("Agent (Execution - Parent IT Debug): LLM fix response contained no actionable edits. Retrying tests.")
                                else:
                                    self.io.tool_output(f"Agent (Execution - Parent IT Debug): Applied LLM suggested fix. Touched: {applied_fix_info['applied_paths']}, Created: {applied_fix_info['created_paths']}")
                                    for new_file_path_rel in applied_fix_info["created_paths"]:
                                        new_file_path_abs = self.abs_root_path(new_file_path_rel)
                                        if new_file_path_abs not in self.abs_fnames:
                                            self.io.add_abs_fname(new_file_path_abs)
                                    self.check_added_files()

                                # Re-run tests for this parent task after applying the fix
                                self.io.tool_output(f"Agent (Execution - Parent IT Debug): Re-running integration tests for parent task '{task_description}' after fix attempt {parent_integration_attempt}...")
                                current_test_output, test_success_after_fix = self._run_tests(is_integration_test=True)
                                if test_success_after_fix:
                                    self.io.tool_output(f"Agent (Execution - Parent IT Debug): Integration tests for '{task_description}' PASSED after fix.")
                                    parent_integration_fixed = True
                                else:
                                    self.io.tool_error(f"Agent (Execution - Parent IT Debug): Integration tests for '{task_description}' STILL FAILED after fix attempt. Output:\n{current_test_output}")
                                    if parent_integration_attempt >= max_parent_integration_retries:
                                        self.io.tool_error(f"Agent (Execution - Parent IT Debug): Max retries reached for parent task '{task_description}'.")
                            # End of while loop for parent integration debugging

                            if parent_integration_fixed:
                                task_completed_successfully = True
                            else:
                                self.io.tool_error(f"Agent (Execution): Failed to fix integration tests for parent task '{task_description}' after all attempts. Marking as failed.")
                                task_completed_successfully = False
                            # End of debugging logic for parent task integration failure

                    else:
                        self.io.tool_output(f"Agent (Execution): No specific integration tests defined for parent task '{task_description}'. Assuming success after sub-tasks completed.")
                        task_completed_successfully = True # Success if all sub-tasks passed and no specific integration tests to fail
                else: # Not all sub-tasks completed
                    task_completed_successfully = False # Parent task fails if any sub-task fails

            else:
                # === Atomic Task (or task treated as atomic due to planning depth/structure) ===
                # This task either was marked atomic, or has no sub_tasks from planning phase.
                atomic_task_type = "Atomic Task" if task.get("is_atomic", True) else "Task (treated as atomic)"
                self.io.tool_output(f"Agent (Execution): Task '{task_description}' is being executed as: {atomic_task_type}.")
                
                task_to_execute = {
                    "id": task_id,
                    "description": task_description,
                    "unit_tests": task.get("unit_tests", []),
                    "type": atomic_task_type
                }
                task_completed_successfully = self._execute_task_recursively(task_to_execute)

            # Record result for the task (which was at the level of original "major deliverable")
            if task_completed_successfully:
                self.completed_deliverables.append(task_id)
                self.io.tool_output(f"Agent (Execution): Successfully completed task: '{task_description}' (ID: {task_id})")
            else:
                self.failed_deliverables.append(task_id)
                self.io.tool_error(f"Agent (Execution): Failed to complete task: '{task_description}' (ID: {task_id})")
                # Optional: Decide if we should stop all execution if one task fails
                # For now, continue to the next task

        self.io.tool_output("\nAgent (Execution): All tasks processed.")
        self.current_phase = "integration_testing" # This phase will run overall_integration_tests
        return self.run()

    def run_integration_testing_phase(self):
        self.check_added_files() # Ensure file list is current
        """Handles the final integration testing phase."""
        self.io.tool_output("Agent (Integration Testing): Running integration tests...")

        overall_tests = self.plan.get("overall_integration_tests", []) if self.plan else []

        if not overall_tests:
            self.io.tool_output("Agent (Integration Testing): No overall integration tests defined in the plan. Skipping.")
            self.current_phase = "reporting"
            return self.run()

        max_retries = self.io.agent_max_integration_test_retries
        attempt = 0
        integration_tests_passed = False

        while attempt < max_retries and not integration_tests_passed:
            attempt += 1
            self.io.tool_output(f"Agent (Integration Testing): Attempt {attempt}/{max_retries}...")
            
            test_output, test_success = self._run_tests(is_integration_test=True)

            if test_success:
                self.io.tool_output("Agent (Integration Testing): All integration tests passed!")
                integration_tests_passed = True
                break
            else:
                self.io.tool_error(f"Agent (Integration Testing): Integration tests failed. Output:\\n{test_output}")
                if attempt >= max_retries:
                    self.io.tool_error("Agent (Integration Testing): Max retries reached for integration tests.")
                    break

                self.io.tool_output("Agent (Integration Testing): Attempting to fix integration test failures...")

                plan_overview = json.dumps(self.plan, indent=2) if self.plan else "No plan available."
                failed_test_description = "Integration tests failed. See output above."
                
                it_web_search_query = f"Fix integration test error: {test_output[:200]}"
                # === Web search for integration testing debug ===
                it_web_results_str = None
                if self.search_enhancer and self.args.agent_web_search != "never":
                    search_now_flag = False
                    if self.args.agent_web_search == "always" or self.args.agent_web_search == "on_demand": # Simplified for debug
                        search_now_flag = True
                            
                    if search_now_flag:
                        self.io.tool_output(f"Agent (Integration Testing - Debug - Web Search): Using Browser-Use for: {it_web_search_query}")
                        try:
                            it_web_results_str = self.search_enhancer.perform_browser_task(it_web_search_query)
                            if it_web_results_str:
                                self.io.tool_output(f"Agent (Integration Testing - Debug - Web Search): Found relevant information.\n{it_web_results_str[:300]}...")
                            else: self.io.tool_output("Agent (Integration Testing - Debug - Web Search): No useful extracts.")
                        except Exception as e:
                            self.io.tool_error(f"Agent (Integration Testing - Debug - Web Search): Error: {e}")
                # === End Web search for integration testing debug ===
                it_web_context_str = it_web_results_str if it_web_results_str else "No web search performed or no results found."

                self.check_added_files()
                code_context_for_it_debug = self.get_context_for_prompts(show_abs_paths=True)
                llm_fix_response = None

                if self.enable_planner_executor_arch:
                    self.io.tool_output("Agent (Integration Testing): Using Planner/Executor debugging flow.")
                    analysis_prompt_content = prompts.agent_analyze_integration_error_system.format(
                        plan_overview=plan_overview,
                        failed_test_description=failed_test_description,
                        test_output_and_errors=test_output,
                        code_context=code_context_for_it_debug,
                        web_search_context=it_web_context_str
                    )
                    analysis_messages = [{"role": "system", "content": analysis_prompt_content}]
                    planner_analysis_json = self._get_llm_response(analysis_messages, expecting_json=True, model_role="planner")

                    if planner_analysis_json and isinstance(planner_analysis_json, dict) and all(k in planner_analysis_json for k in ["error_analysis", "fix_plan", "target_files_and_lines_suggestion"]):
                        
                        error_analysis = planner_analysis_json["error_analysis"]
                        fix_plan = planner_analysis_json["fix_plan"]
                        suggestion = planner_analysis_json["target_files_and_lines_suggestion"]
                        self.io.tool_output(f"Agent (Planner Analysis - Integration): Error: {error_analysis}. Plan: {fix_plan}. Suggestion: {suggestion}")

                        general_task_desc_for_fix = f"Fix integration test failures for overall task: {self.clarified_task if self.clarified_task else self.initial_task}"
                        implement_fix_prompt_content = prompts.agent_implement_fix_plan_system.format(
                            task_description=general_task_desc_for_fix,
                            failed_code_attempt="Multiple files may have changed, see context.",
                            test_output_and_errors=test_output,
                            error_analysis_from_planner=error_analysis,
                            fix_plan_from_planner=fix_plan,
                            target_files_and_lines_suggestion_from_planner=suggestion,
                            file_context=code_context_for_it_debug 
                        )
                        implement_fix_messages = [{"role": "system", "content": implement_fix_prompt_content}]
                        llm_fix_response = self._get_llm_response(implement_fix_messages, model_role="executor")
                    else:
                        self.io.tool_warning("Agent (Planner Analysis - Integration): Failed to get valid analysis from Planner LLM.")
                else:
                    self.io.tool_output("Agent (Integration Testing): Using legacy debugging flow.")
                    legacy_debug_prompt = prompts.agent_integration_debugging_system.format(
                        plan_overview=plan_overview,
                        failed_test_description=failed_test_description,
                        test_output_and_errors=test_output,
                        web_search_context=it_web_context_str,
                        code_context=code_context_for_it_debug,
                    )
                    legacy_debug_messages = [{"role": "system", "content": legacy_debug_prompt}]
                    llm_fix_response = self._get_llm_response(legacy_debug_messages, model_role="planner")

                if not llm_fix_response:
                    self.io.tool_error("Agent (Integration Testing): LLM failed to provide a fix or analysis was invalid. Retrying tests as is (if attempts left).")
                    continue

                applied_fix_info = self._parse_and_apply_edits(llm_fix_response)
                if not applied_fix_info["applied_paths"] and not applied_fix_info["created_paths"]:
                    self.io.tool_error("Agent (Integration Testing): LLM fix response contained no actionable edits. Retrying tests.")
                else:
                    self.io.tool_output(f"Agent (Integration Testing): Applied LLM suggested fix. Touched: {applied_fix_info['applied_paths']}, Created: {applied_fix_info['created_paths']}")
                    for new_file_path_rel in applied_fix_info["created_paths"]:
                        new_file_path_abs = self.abs_root_path(new_file_path_rel)
                        if new_file_path_abs not in self.abs_fnames:
                             self.io.add_abs_fname(new_file_path_abs)
                    self.check_added_files()
        
        if not integration_tests_passed:
            self.io.tool_error("Agent (Integration Testing): Failed to pass integration tests after all attempts.")
            self.integration_tests_final_status = "FAILED"
        else:
            self.integration_tests_final_status = "PASSED"
        
        self.current_phase = "reporting"
        return self.run()

    def run_reporting_phase(self):
        """Handles the final reporting phase."""
        self.io.tool_output("Agent (Reporting): Task complete. Generating final report...")

        # Close browser if SearchEnhancer was used
        if self.search_enhancer and hasattr(self.search_enhancer, 'close_browser'):
            self.io.tool_output("Agent (Reporting): Closing browser used by SearchEnhancer...")
            try:
                self.search_enhancer.close_browser()
            except Exception as e:
                self.io.tool_error(f"Agent (Reporting): Error closing browser: {e}")

        summary = ["### Agent Task Summary ###"]
        summary.append(f"**Initial Task:** {self.initial_task}")
        if hasattr(self, 'clarified_task') and self.clarified_task and self.clarified_task != self.initial_task:
            summary.append(f"**Clarified Task:** {self.clarified_task}")

        if not self.plan or "tasks" not in self.plan or not self.plan["tasks"]:
            summary.append("\nNo tasks were planned or executed.")
        else:
            summary.append("\n**Execution Report:**")
            num_tasks = len(self.plan["tasks"])
            for i, task in enumerate(self.plan["tasks"]):
                task_id = task["id"]
                task_desc = task["description"]
                status = "Unknown"
                if task_id in self.completed_deliverables:
                    status = "Completed Successfully"
                elif task_id in self.failed_deliverables:
                    status = "Failed"
                # Could also check if it was skipped or not reached if we add such tracking

                summary.append(f"\n**Task {i+1}/{num_tasks}: {task_desc}** (ID: {task_id}) - Status: {status}")

                if task.get("unit_tests"):
                     summary.append(f"  Unit Tests: {'Considered (see task status)' if status != 'Unknown' else 'Not Run'}")

        summary.append(f"\n**Overall Integration Test Status:** {self.integration_tests_final_status}")

        # Files changed (from AgentCoder's internal tracking)
        if self.agent_touched_files_rel:
            summary.append("\n**Files touched by agent (created or modified):**")
            for fname_rel in sorted(list(self.agent_touched_files_rel)):
                summary.append(f"  - {fname_rel}")
        else:
            summary.append("\nNo files appear to have been created or modified by the agent in this session.")
        
        summary.append("\nFor detailed changes, please review the git history if commits were made, or the diffs if shown.")

        report = "\n".join(summary)
        self.io.tool_output(report)
        
        self.current_phase = "idle" 

        if self.from_coder:
            self.io.tool_output(f"Switching back to {self.from_coder.coder_name} coder.")
            # Basic switch back. A more robust solution would restore the exact state.
            # This relies on Coder.create to reinitialize from_coder appropriately.
            # We need to pass essential state like fnames, read_only_fnames, history etc.
            # The `from_coder` in SwitchCoder handles passing the instance,
            # and Coder.create uses its attributes.
            
            # Prepare kwargs to restore essential parts of the from_coder's state
            # This is a simplified restoration. A full restoration might need a dedicated
            # get_state_kwargs() method on all coders.
            kwargs_for_switch_back = {
                "fnames": list(self.from_coder.abs_fnames),
                "read_only_fnames": list(self.from_coder.abs_read_only_fnames),
                "done_messages": self.from_coder.done_messages,
                "cur_messages": self.from_coder.cur_messages,
                 # Add other relevant state that from_coder might need
            }
            if hasattr(self.from_coder, 'original_kwargs'):
                 kwargs_for_switch_back.update(self.from_coder.original_kwargs)


            raise SwitchCoder(type(self.from_coder), **kwargs_for_switch_back)
        else:
            self.io.tool_warning("Agent: No previous coder to switch back to. Agent remains idle.")
            return "Agent task finished. No previous coder to restore."

        return "Agent task finished. Switched back to previous coder."

    def run_outputting_plan_phase(self):
        """Outputs the generated plan and tests, then stops."""
        self.io.tool_output("Agent (Outputting Plan): Final plan and test strategy:")
        
        final_output = {
            "initial_task": self.initial_task,
            "clarified_task": getattr(self, 'clarified_task', self.initial_task),
            "plan": self.plan, # Contains hierarchical tasks and their specific tests
            "suggested_test_command": self.agent_test_command
        }
        
        try:
            output_json = json.dumps(final_output, indent=2, default=str)
            # This output should go to stdout for programmatic consumption
            print(output_json) 
            self.io.tool_output("Plan output to stdout complete.")
        except TypeError as e:
            self.io.tool_error(f"Could not serialize final plan to JSON: {e}")
            # Fallback or error indication
            print(json.dumps({"error": "Could not serialize plan", "details": str(e)}, indent=2))

        self.current_phase = "idle" # Or a specific "halted_after_plan_output" state
        # Return a message indicating completion for the main loop
        return "Agent finished: Plan and tests generated and outputted. Execution not performed as per --agent-output-plan-only."

    # Helper methods
    def _get_llm_response(self, messages, expecting_json=False, model_role="planner"):
        """ Helper to send messages to the chosen model and get the response content. """
        attempt_count = 0
        max_attempts = 10 if expecting_json else 2 # Allow more retries for JSON parsing

        active_model = self.main_model # Default
        if self.enable_planner_executor_arch:
            if model_role == "planner" and self.planner_llm:
                active_model = self.planner_llm
            elif model_role == "executor" and self.executor_llm:
                active_model = self.executor_llm
            else: # Fallback
                active_model = self.main_model
        else:
            active_model = self.main_model

        if not active_model:
            self.io.tool_error("AgentCoder: Critical error - no active model determined for LLM call.")
            return None

        while attempt_count < max_attempts:
            attempt_count += 1
            try:
                self.partial_response_content = ""
                self.partial_response_function_call = dict()
                self.message_cost = 0.0 

                # For Gemini models, add a user message if we only have a system message
                if active_model.name.startswith("gemini/") and len(messages) == 1 and messages[0].get("role") == "system":
                    messages.append({"role": "user", "content": "Please provide your response based on the system instructions above."})
                
                for chunk_content in self.send(messages, model=active_model):
                    pass  # Just consume the stream

                response_content = self.partial_response_content

                if expecting_json:
                    # Strip markdown fences if present
                    stripped_response = response_content.strip()
                    if stripped_response.startswith("```") and stripped_response.endswith("```"):
                        first_newline = stripped_response.find('\n')
                        if first_newline != -1:
                            stripped_response = stripped_response[first_newline + 1:-3].strip()
                        else:
                            stripped_response = stripped_response[3:-3].strip()
                    
                    try:
                        parsed_json = json.loads(stripped_response)
                        return parsed_json
                    except json.JSONDecodeError:
                        # Try finding specific ```json block
                        match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(1).strip())
                            except json.JSONDecodeError:
                                pass
                    
                    # If we are here, JSON parsing failed
                    if attempt_count < max_attempts:
                        self.io.tool_warning(f"Failed to parse JSON response (attempt {attempt_count}/{max_attempts}). Retrying...")
                        continue
                    else:
                        self.io.tool_error("Failed to parse JSON after all attempts.")
                        return None
                else:
                    # Not expecting JSON, so we return the raw content
                    return response_content

            except Exception as e:
                self.io.tool_error(f"Error communicating with LLM (attempt {attempt_count}/{max_attempts}): {e}")
                if attempt_count >= max_attempts:
                    return None
        
        return None

    def _run_shell_command(self, command):
        """ Helper to run a shell command and return output, error, and exit code. """
        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo.root if self.repo else None # Run in repo root if available
            )
            return process.stdout, process.stderr, process.returncode
        except Exception as e:
            self.io.tool_error(f"Error running shell command '{command}': {e}")
            return None, str(e), -1 # Indicate error with -1 exit code

    def _delegate_to_coder(self, task_message, use_executor_model=True):
        """
        Delegates a task to the existing coder infrastructure.
        Returns True if successful, False otherwise.
        """
        try:
            # Choose the model for the delegated coder
            if use_executor_model and self.enable_planner_executor_arch and self.executor_llm:
                model_for_delegation = self.executor_llm
            else:
                model_for_delegation = self.main_model

            # Import here to avoid circular imports
            from aider.coders.editblock_coder import EditBlockCoder
            
            # Create a temporary coder with the same context
            delegated_coder = EditBlockCoder(
                main_model=model_for_delegation,
                io=self.io,
                args=self.args,
                repo=self.repo,
                fnames=list(self.abs_fnames),
                read_only_fnames=list(self.abs_read_only_fnames)
            )
            
            # Transfer essential state
            delegated_coder.abs_fnames = self.abs_fnames.copy()
            delegated_coder.abs_read_only_fnames = self.abs_read_only_fnames.copy()
            
            # Run the coder for one iteration with the task message
            response = delegated_coder.run(with_message=task_message)
            
            # Update our file tracking based on what the delegated coder did
            new_files = delegated_coder.abs_fnames - self.abs_fnames
            modified_files = delegated_coder.abs_fnames & self.abs_fnames
            
            # Update our state
            self.abs_fnames = delegated_coder.abs_fnames.copy()
            self.abs_read_only_fnames = delegated_coder.abs_read_only_fnames.copy()
            
            # Track changes for reporting
            for new_file in new_files:
                rel_path = self.get_rel_fname(new_file)
                self.agent_touched_files_rel.add(rel_path)
            
            for modified_file in modified_files:
                rel_path = self.get_rel_fname(modified_file)
                self.agent_touched_files_rel.add(rel_path)
            
            # Consider successful if no exceptions and response was generated
            return response is not None
            
        except Exception as e:
            self.io.tool_error(f"Error delegating to coder: {e}")
            return False

    def _decompose_task_recursively(self, task_description, current_depth, max_depth, parent_id_prefix):
        """
        Recursively decomposes a task into sub-tasks up to max_depth.
        Returns a dictionary representing the task and its decomposed sub-tasks.
        """
        task_id = f"{parent_id_prefix}_d{current_depth}"

        # If max_depth is reached, the task is considered atomic regardless of its content.
        if current_depth >= max_depth:
            self.io.tool_output(f"Agent (Planning): Task '{task_description[:100]}...' at depth {current_depth} (max_depth {max_depth} reached). Treating as atomic.")
            return {
                "id": task_id,
                "description": task_description, # Keep the original description
                "is_atomic": True,
                "sub_tasks": [],
                "depth": current_depth,
                "unit_tests": [],
                "integration_tests": [] 
            }

        # The prompt now incorporates current_depth and max_depth to guide the LLM
        decompose_prompt_text = prompts.agent_recursive_decompose_task_system.format(
            task_description=task_description,
            current_depth=current_depth,
            max_depth=max_depth
        )
        messages_decompose = [{"role": "system", "content": decompose_prompt_text}]
        
        # _get_llm_response will parse the JSON: {"is_atomic": bool, "sub_tasks": [str, ...]}
        llm_decomposition_result = self._get_llm_response(messages_decompose, expecting_json=True, model_role="planner")

        if llm_decomposition_result is None or \
           not isinstance(llm_decomposition_result, dict) or \
           'is_atomic' not in llm_decomposition_result or \
           'sub_tasks' not in llm_decomposition_result or \
           not isinstance(llm_decomposition_result['is_atomic'], bool) or \
           not isinstance(llm_decomposition_result['sub_tasks'], list):
            self.io.tool_warning(
                f"Agent (Planning): Failed to get valid decomposition for task: '{task_description}' "
                f"at depth {current_depth}. LLM Response: {llm_decomposition_result}. Treating as atomic."
            )
            return {
                "id": task_id,
                "description": task_description,
                "is_atomic": True,
                "sub_tasks": [],
                "depth": current_depth,
                "unit_tests": [],      # Initialize, will be filled by test design phase
                "integration_tests": [] # Initialize, will be filled by test design phase
            }

        is_llm_atomic = llm_decomposition_result['is_atomic']
        raw_sub_task_descs = llm_decomposition_result['sub_tasks']

        # Determine final atomicity based on LLM and depth rules (LLM was already instructed about depth)
        final_is_atomic = is_llm_atomic
        if not final_is_atomic and not raw_sub_task_descs:
            self.io.tool_warning(f"Agent (Planning): Task '{task_description}' at depth {current_depth} marked non-atomic by LLM but no sub-tasks provided. Treating as atomic.")
            final_is_atomic = True
        elif current_depth >= max_depth and not final_is_atomic:
            self.io.tool_output(f"Agent (Planning): Task '{task_description}' at depth {current_depth} (max_depth reached). Forcing to atomic even if LLM suggested sub-tasks.")
            final_is_atomic = True # Enforce atomicity if max_depth is met, overriding LLM if it still gave sub-tasks
        elif final_is_atomic:
            self.io.tool_output(f"Agent (Planning): Task '{task_description}' at depth {current_depth} is atomic (LLM decision or depth limit). No further decomposition.")
            raw_sub_task_descs = [] # Ensure no sub-tasks if atomic

        if final_is_atomic:
            return {
                "id": task_id,
                "description": task_description,
                "is_atomic": True,
                "sub_tasks": [],
                "depth": current_depth,
                "unit_tests": [],
                "integration_tests": [] 
            }

        # If not atomic, proceed with recursive decomposition of sub-tasks
        decomposed_sub_tasks = []
        valid_sub_tasks_found = False
        for i, sub_task_desc_raw in enumerate(raw_sub_task_descs):
            if not isinstance(sub_task_desc_raw, str) or not sub_task_desc_raw.strip():
                self.io.tool_warning(
                    f"Agent (Planning): Sub-task for task '{task_description}' "
                    f"is not a valid string or is empty: '{sub_task_desc_raw}'. Skipping."
                )
                continue
            
            sub_task_desc = sub_task_desc_raw.strip()
            self.io.tool_output(f"Agent (Planning): Recursively decomposing sub-task '{sub_task_desc}' for '{task_description}' (next depth {current_depth + 1})")
            
            new_sub_task_parent_id_prefix = f"{task_id}_st{i+1}"
            sub_task_node = self._decompose_task_recursively(
                task_description=sub_task_desc,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                parent_id_prefix=new_sub_task_parent_id_prefix 
            )
            decomposed_sub_tasks.append(sub_task_node)
            valid_sub_tasks_found = True

        if not valid_sub_tasks_found:
            self.io.tool_warning(f"Agent (Planning): Task '{task_description}' at depth {current_depth} was non-atomic, but no valid sub-tasks resulted from recursion. Treating as atomic.")
            return {
                "id": task_id,
                "description": task_description,
                "is_atomic": True,
                "sub_tasks": [],
                "depth": current_depth,
                "unit_tests": [],
                "integration_tests": []
            }

        return {
            "id": task_id,
            "description": task_description,
            "is_atomic": False, # Parent is not atomic if it has valid sub-tasks
            "sub_tasks": decomposed_sub_tasks,
            "depth": current_depth,
            "unit_tests": [], 
            "integration_tests": [] # For this parent task, to test its sub_tasks' integration
        }

    def _execute_task_recursively(self, task_node, parent_description=None):
        """
        Executes a task node, which might be atomic or a parent with sub-tasks.
        Returns True if successful, False otherwise.
        """
        task_id = task_node["id"]
        task_desc = task_node["description"]
        is_atomic = task_node.get("is_atomic", True) # Default to atomic if not specified
        sub_tasks = task_node.get("sub_tasks", [])

        full_task_desc_for_logging = f"'{parent_description} -> {task_desc}'" if parent_description else f"'{task_desc}'"
        self.io.tool_output(f"Agent (Execution - Recursive): Starting task {full_task_desc_for_logging} (ID: {task_id})")

        if is_atomic or not sub_tasks:
            # Task is atomic or has no sub-tasks, execute as atomic
            # Prepare the task_to_execute dict for _execute_atomic_task
            atomic_task_spec = {
                "id": task_id,
                "description": task_desc,
                "unit_tests": task_node.get("unit_tests", []),
                "type": "Atomic (or leaf) Task"
            }
            self.io.tool_output(f"Agent (Execution - Recursive): Task {full_task_desc_for_logging} is atomic/leaf. Executing directly.")
            return self._execute_atomic_task(atomic_task_spec, parent_md_description=parent_description)
        else:
            # Task is a parent with sub-tasks
            self.io.tool_output(f"Agent (Execution - Recursive): Task {full_task_desc_for_logging} is a parent. Processing its {len(sub_tasks)} sub-tasks.")
            all_sub_tasks_completed_successfully = True
            for i, sub_task_node_item in enumerate(sub_tasks):
                self.io.tool_output(f"Agent (Execution - Recursive): Starting sub-task {i+1}/{len(sub_tasks)} of {full_task_desc_for_logging} (ID: {sub_task_node_item['id']})")
                sub_task_success = self._execute_task_recursively(sub_task_node_item, parent_description=task_desc)
                if not sub_task_success:
                    self.io.tool_error(f"Agent (Execution - Recursive): Sub-task '{sub_task_node_item['description']}' (ID: {sub_task_node_item['id']}) of {full_task_desc_for_logging} FAILED.")
                    all_sub_tasks_completed_successfully = False
                    break 
            
            if not all_sub_tasks_completed_successfully:
                self.io.tool_error(f"Agent (Execution - Recursive): Parent task {full_task_desc_for_logging} FAILED due to sub-task failure.")
                return False

            # All sub-tasks completed successfully, now run integration tests for this parent task
            self.io.tool_output(f"Agent (Execution - Recursive): All sub-tasks of {full_task_desc_for_logging} completed successfully.")
            parent_integration_tests = task_node.get("integration_tests", [])
            if parent_integration_tests:
                self.io.tool_output(f"Agent (Execution - Recursive): Running integration tests for parent task {full_task_desc_for_logging}. Tests: {json.dumps(parent_integration_tests)}")
                # _run_tests uses the general test command. The specific tests are for LLM guidance if debugging needed.
                test_output, test_success = self._run_tests(is_integration_test=True)
                if test_success:
                    self.io.tool_output(f"Agent (Execution - Recursive): Integration tests for parent task {full_task_desc_for_logging} PASSED.")
                    return True # Parent task success
                else:
                    self.io.tool_error(f"Agent (Execution - Recursive): Integration tests for parent task {full_task_desc_for_logging} FAILED. Output:\\n{test_output}")

                    # Begin debugging loop for parent task integration failure in _execute_task_recursively
                    max_recursive_integration_retries = MAX_INTEGRATION_TEST_RETRIES
                    recursive_integration_attempt = 0
                    recursive_integration_fixed = False
                    current_recursive_test_output = test_output

                    while recursive_integration_attempt < max_recursive_integration_retries and not recursive_integration_fixed:
                        recursive_integration_attempt += 1
                        self.io.tool_output(f"Agent (Execution - Recursive IT Debug): Attempt {recursive_integration_attempt}/{max_recursive_integration_retries} to fix integration for {full_task_desc_for_logging}...")

                        self.check_added_files()
                        code_context_recursive_debug = self.get_context_for_prompts(show_abs_paths=True)
                        # Plan overview for this context should ideally be the specific task_node and its lineage
                        # For simplicity, using the whole plan. Could be refined to pass task_node for more focused prompt.
                        plan_overview_recursive_debug = json.dumps(self.plan, indent=2)
                        failed_test_desc_recursive = f"Integration tests for task {full_task_desc_for_logging} failed."
                        
                        recursive_it_web_query = f"Fix integration test error for {task_desc}: {current_recursive_test_output[:200]}"
                        # === Web search for recursive IT debug ===
                        recursive_it_web_results_str = None
                        if self.search_enhancer and self.args.agent_web_search != "never":
                            search_now_flag = False
                            if self.args.agent_web_search == "always" or self.args.agent_web_search == "on_demand": # Simplified for debug
                                search_now_flag = True

                            if search_now_flag:
                                self.io.tool_output(f"Agent (Exec - Recursive IT Debug - Web Search): Using Browser-Use for: {recursive_it_web_query}")
                                try:
                                    recursive_it_web_results_str = self.search_enhancer.perform_browser_task(recursive_it_web_query)
                                    if recursive_it_web_results_str:
                                        self.io.tool_output(f"Agent (Exec - Recursive IT Debug - Web Search): Found relevant information.\n{recursive_it_web_results_str[:300]}...")
                                    else: self.io.tool_output("Agent (Exec - Recursive IT Debug - Web Search): No useful extracts.")
                                except Exception as e:
                                    self.io.tool_error(f"Agent (Exec - Recursive IT Debug - Web Search): Error: {e}")
                        # === End Web search for recursive IT debug ===
                        recursive_it_web_context = recursive_it_web_results_str if recursive_it_web_results_str else "No web search."

                        llm_fix_response_recursive = None
                        if self.enable_planner_executor_arch:
                            analysis_prompt = prompts.agent_analyze_integration_error_system.format(
                                plan_overview=plan_overview_recursive_debug,
                                failed_test_description=failed_test_desc_recursive,
                                test_output_and_errors=current_recursive_test_output,
                                code_context=code_context_recursive_debug,
                                web_search_context=recursive_it_web_context
                            )
                            analysis_msgs = [{"role": "system", "content": analysis_prompt}]
                            planner_analysis = self._get_llm_response(analysis_msgs, expecting_json=True, model_role="planner")

                            if planner_analysis and isinstance(planner_analysis, dict) and all(k in planner_analysis for k in ["error_analysis", "fix_plan"]):
                                error_analysis = planner_analysis["error_analysis"]
                                fix_plan = planner_analysis["fix_plan"]
                                suggestion = planner_analysis.get("target_files_and_lines_suggestion", "No specific suggestion.")
                                self.io.tool_output(f"Agent (Planner - Recursive IT): Analysis: {error_analysis}. Plan: {fix_plan}. Suggestion: {suggestion}")

                                implement_prompt = prompts.agent_implement_fix_plan_system.format(
                                    task_description=f"Fix integration issues for task: {task_desc} (parent: {parent_description})",
                                    failed_code_attempt="Context contains current state.",
                                    test_output_and_errors=current_recursive_test_output,
                                    error_analysis_from_planner=error_analysis,
                                    fix_plan_from_planner=fix_plan,
                                    target_files_and_lines_suggestion_from_planner=suggestion,
                                    file_context=code_context_recursive_debug
                                )
                                implement_msgs = [{"role": "system", "content": implement_prompt}]
                                llm_fix_response_recursive = self._get_llm_response(implement_msgs, model_role="executor")
                            else:
                                self.io.tool_warning("Agent (Planner - Recursive IT): Failed to get valid analysis.")
                        else: # Legacy flow
                            debug_prompt = prompts.agent_integration_debugging_system.format(
                                plan_overview=plan_overview_recursive_debug,
                                failed_test_description=failed_test_desc_recursive,
                                test_output_and_errors=current_recursive_test_output,
                                web_search_context=recursive_it_web_context,
                                code_context=code_context_recursive_debug
                            )
                            debug_msgs = [{"role": "system", "content": debug_prompt}]
                            llm_fix_response_recursive = self._get_llm_response(debug_msgs, model_role="planner")

                        if not llm_fix_response_recursive:
                            self.io.tool_error("Agent (Exec - Recursive IT Debug): LLM failed to provide fix. Retrying tests.")
                            current_recursive_test_output, test_success_after_failed_fix = self._run_tests(is_integration_test=True)
                            if test_success_after_failed_fix:
                                recursive_integration_fixed = True
                            continue

                        applied_fix = self._parse_and_apply_edits(llm_fix_response_recursive)
                        if not applied_fix["applied_paths"] and not applied_fix["created_paths"]:
                            self.io.tool_error("Agent (Exec - Recursive IT Debug): LLM fix had no edits. Retrying tests.")
                        else:
                            self.io.tool_output(f"Agent (Exec - Recursive IT Debug): Applied fix. Touched: {applied_fix['applied_paths']}, Created: {applied_fix['created_paths']}")
                            for new_file in applied_fix["created_paths"]:
                                self.io.add_abs_fname(self.abs_root_path(new_file))
                            self.check_added_files()

                        self.io.tool_output(f"Agent (Exec - Recursive IT Debug): Re-running integration tests for {full_task_desc_for_logging} after fix attempt...")
                        current_recursive_test_output, test_success_after_fix = self._run_tests(is_integration_test=True)
                        if test_success_after_fix:
                            self.io.tool_output(f"Agent (Exec - Recursive IT Debug): Tests PASSED for {full_task_desc_for_logging}.")
                            recursive_integration_fixed = True
                        else:
                            self.io.tool_error(f"Agent (Exec - Recursive IT Debug): Tests STILL FAILED for {full_task_desc_for_logging}. Output:\n{current_recursive_test_output}")
                            if recursive_integration_attempt >= max_recursive_integration_retries:
                                self.io.tool_error(f"Agent (Exec - Recursive IT Debug): Max retries for {full_task_desc_for_logging}.")
                    
                    if recursive_integration_fixed:
                        return True # Parent task success after debugging
                    else:
                        self.io.tool_error(f"Agent (Execution - Recursive): Failed to fix integration tests for {full_task_desc_for_logging} after all attempts.")
                        return False # Parent task failed integration after all attempts

            else:
                self.io.tool_output(f"Agent (Execution - Recursive): No specific integration tests for parent task {full_task_desc_for_logging}. Assuming success.")
                return True # Parent task success (all sub-tasks passed, no integration tests to fail)

    def _execute_atomic_task(self, task_to_execute, parent_md_description=None):
        """Executes an atomic task using delegation to existing coder infrastructure."""
        if not task_to_execute or not task_to_execute.get("description"):
            self.io.tool_warning("AgentCoder: Atomic task has no description. Skipping.")
            return False

        task_description = task_to_execute["description"]
        self.io.tool_output(f"AgentCoder: Executing atomic task: {task_description}")

        # Prepare task message with context
        task_message = f"Please implement the following task:\n\n{task_description}"
        
        if parent_md_description:
            task_message += f"\n\nThis task is part of the larger deliverable: {parent_md_description}"

        # Use delegation instead of custom edit parsing
        success = self._delegate_to_coder(task_message, use_executor_model=True)
        
        if success:
            self.io.tool_output("AgentCoder: Atomic task completed successfully.")
            return True
        else:
            self.io.tool_error("AgentCoder: Atomic task failed.")
            return False

    # Helper methods start here
    def _write_file_ensure_path(self, file_path_abs_str: str, content: str) -> bool:
        """
        Writes content to a file, ensuring parent directories exist.
        Returns True on success, False on failure.
        """
        try:
            path_obj = Path(file_path_abs_str)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.io.write_text(file_path_abs_str, content) # Use io.write_text for consistency
            return True
        except Exception as e:
            self.io.tool_error(f"AgentCoder: Error writing file {file_path_abs_str}: {e}")
            return False

    def _confirm_action(self, prompt_message: str, default_response: str = 'y') -> bool:
        """
        Asks the user for confirmation for an action.
        Returns True if confirmed, False otherwise.
        Handles headless mode by auto-confirming if agent_auto_approve is True.
        """
        # In headless mode (which implies auto-approve if agent_auto_approve is True) 
        # or if auto_approve is explicitly set True outside of headless mode.
        should_auto_confirm = False
        if self.is_headless:
            if hasattr(self.args, 'agent_auto_approve') and self.args.agent_auto_approve:
                should_auto_confirm = True
        elif hasattr(self.args, 'agent_auto_approve') and self.args.agent_auto_approve:
            should_auto_confirm = True
            
        if should_auto_confirm:
            self.io.tool_output(f"Agent (Auto-Confirm): Action '{prompt_message}' auto-confirmed (headless with auto-approve, or auto-approve active).")
            return True
        return self.io.confirm_ask(prompt_message, default=default_response)

    def _run_tests(self, is_integration_test=False):
        """
        Runs tests and returns their output and success status.
        Uses self.agent_test_command if set, otherwise io.test_command or io.integration_test_command.
        """
        test_command = None
        if self.agent_test_command:
            test_command = self.agent_test_command
            self.io.tool_output(f"Using LLM suggested test command: {test_command}")
        elif is_integration_test:
            test_command = self.io.integration_test_command
            self.io.tool_output(f"Using configured integration test command: {test_command}")
        else: # Unit test
            test_command = self.io.test_command
            self.io.tool_output(f"Using configured unit test command: {test_command}")

        if not test_command:
            self.io.tool_warning("No test command configured. Skipping test run.")
            return "No test command configured.", False

        self.io.tool_output(f"Running: {test_command}")
        output, error, exit_code = self._run_shell_command(test_command)
        
        full_output = f"STDOUT:\n{output}\nSTDERR:\n{error}\nEXIT CODE: {exit_code}"
        success = (exit_code == 0)
        
        return full_output, success
