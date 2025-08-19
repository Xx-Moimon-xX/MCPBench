import dspy
from pydantic import BaseModel, Field
from langProBe.program_utils import (
    call_lm, 
    build_init_messages, 
    build_messages,
    response_parsing,
    mcp_calling,
    ProcessManager,
    MCPCall,
    MCPCallList, 
    build_system_content
)
import time
from langProBe.evaluation_utils import evaluate_final_answer
import langProBe.constants as constants
import logging
import os
from datetime import datetime
import json, csv
from typing import List, Dict, Optional, Tuple


"""
Main file that deals with the program/system logic to get responses.
Functions/classes present: 
- MCPPredict Class: Main class that deals with the program/system logic to get responses.
- Logger created and used in this
- evaluate prediction is also in this.

Uses the program utils file to build the messages, call the LLM, and parse responses and call MCPs.
"""

MCP_SAMPLE_SYSTEM_PROMPT = """
You are a helpful assistant. You are able to answer questions using different tools.  
The content of your available tools begins with ## Available Tools, indicating the collection of usable tools.  
Within the tool collection, each server is identified by ### server_name, where server_name represents the name of the server.  
Under each server, there are multiple tools (tool), and each tool starts with - tool_name, where tool_name is the name of the tool.  
The tool description includes:  
A brief text description outlining the functionality of the tool.  
Detailed information about input parameters, where each parameter includes: parameter name, parameter type, whether it is mandatory, and the purpose or description of the parameter.
"""

class MCP_LM(BaseModel):
    model: str = Field(
        default=None,
        description="The model to use for the MCP program.",
    )
    api_key: str = Field(
        default=None,
        description="The API key for the model.",
    )
    api_base: str = Field(
        default=None,
        description="The API base URL for the model.",
    )
    eval_model: str = Field(
        default=None,
        description="The model to use for evaluation.",
    )

class LangProBeMCPMetaProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = MCP_LM()
        # Controls `forward` behavior
        # Values: 'combined' (default), 'generate_only', 'score_only'
        self.run_mode = 'combined'
        # Root of the run directory for reading saved predictions
        self.run_root_path = None
    def setup_lm(self, lm, api_key=None, api_base=None, eval_model=None):
        self.lm.model = lm
        self.lm.api_key = api_key
        self.lm.api_base = api_base
        self.lm.eval_model = eval_model

    def set_run_mode(self, mode: str):
        if mode not in ('combined', 'generate_only', 'score_only'):
            raise ValueError("run_mode must be 'combined', 'generate_only', or 'score_only'")
        self.run_mode = mode

    # --- Helpers for score_only mode ---
    def _responses_csv_path(self) -> str:
        base = self.log_path or "."
        return os.path.join(base, "response_data", "predictions.csv")

    def _load_saved_predictions_map(self):
        if hasattr(self, "_score_only_saved_map") and self._score_only_saved_map is not None:
            return self._score_only_saved_map
        path = self._responses_csv_path()
        mapping = {}
        if not os.path.exists(path):
            self._score_only_saved_map = mapping
            return mapping
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row.get("id", "")] = row
        self._score_only_saved_map = mapping
        return mapping

    def _get_saved_answer_for_id(self, unique_id: str) -> str:
        mapping = self._load_saved_predictions_map()
        row = mapping.get(str(unique_id))
        return row.get("answer", "") if row else ""

    def program_type(self):
        return "mcp"
    

class MCPPredict(LangProBeMCPMetaProgram, dspy.Module):
    '''
    This is the program/system that is run to get responses. Called MCPPredict.
    '''
    def __init__(self, config, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="mcp_sample", tools_format="formatted"):
        super().__init__()
        self.system_prompt = system_prompt
        self.task_name = task_name
        self.max_steps = max_steps
        self.max_length = 30000
        self.config = config
        self.tools_format = tools_format
        # self.mcps = config.get("mcp_pool")
        # self.mcpserver = self.config["mcp_pool"][0].get("name")

        # Configure run logger
        # Mainly using the run_logger, i don't know what the message logger is for.
        self.run_logger = logging.getLogger('MCPPredictRunLogger')
        self.run_logger.setLevel(logging.DEBUG)

        # Configure message logger
        self.message_logger = logging.getLogger('MCPPredictMessageLogger')
        self.message_logger.setLevel(logging.DEBUG)
        # self.dataset = dataset
        self.log_path = None

        # Create log directory
        os.makedirs('logs', exist_ok=True)
        # self.setup_loggers()

        # Defer building system content to runtime (avoid MCP calls in score_only)
        self._mcps = self.config["mcp_pool"]
        self.system_content = None
        # --- Persistent client cache for MCP tool calls ---
        self.client_cache = None  # Will be set by EvaluateBench if available
        # --------------------------------------------------

    def set_client_cache(self, client_cache):
        self.client_cache = client_cache

    def set_log_path(self, path):
        self.log_path = path
        if self.run_root_path is None:
            self.run_root_path = path
        self.setup_loggers()

    def setup_loggers(self):
        log_dir = self.log_path
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        else:
            return

        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        # Set up run log
        run_log_file = f'{log_dir}/{self.task_name}_run_{timestamp}.log'
        run_handler = logging.FileHandler(run_log_file, encoding='utf-8')
        run_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        run_handler.setFormatter(run_formatter)
        self.run_logger.addHandler(run_handler)

        # Set up message log (skip in score_only mode)
        if getattr(self, 'run_mode', 'combined') != 'score_only':
            message_log_file = f'{log_dir}/{self.task_name}_messages_{timestamp}.jsonl'
            message_handler = logging.FileHandler(message_log_file, encoding='utf-8')
            self.message_logger.addHandler(message_handler)


    def update_log_paths(self, new_log_dir):
        # Ensure the new log directory exists
        os.makedirs(new_log_dir, exist_ok=True)
        
        # Update run logger
        for handler in self.run_logger.handlers[:]:
            self.run_logger.removeHandler(handler)
        
        run_log_file = f'{new_log_dir}/{self.task_name}_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        run_handler = logging.FileHandler(run_log_file, encoding='utf-8')
        run_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        run_handler.setFormatter(run_formatter)
        self.run_logger.addHandler(run_handler)

        # Update message logger
        for handler in self.message_logger.handlers[:]:
            self.message_logger.removeHandler(handler)
        # Skip creating messages file in score_only mode
        if getattr(self, 'run_mode', 'combined') != 'score_only':
            message_log_file = f'{new_log_dir}/{self.task_name}_messages_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
            message_handler = logging.FileHandler(message_log_file, encoding='utf-8')
            self.message_logger.addHandler(message_handler)

    def evaluate_prediction(self, question: str, ground_truth: str, tools_required: List[str], tools_called: List[MCPCall], prediction: str) -> Tuple[bool, Optional[str]]:
        '''
        Main function to evaluate the prediction.
        '''
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.eval_model:
            answer_eval_manager.model = self.lm.eval_model
        else:
            answer_eval_manager.model = self.lm.model

        return evaluate_final_answer(question, ground_truth, tools_required, tools_called, prediction, answer_eval_manager, self.run_logger)

    def log_messages(self, messages, question, success, time_cost, prompt_tokens_cost, completion_tokens_cost):
        log_entry = {
            "question": question,
            "messages": messages,
            "success": success,
            "time_cost": time_cost,
            "prompt_tokens_cost": prompt_tokens_cost,
            "completion_tokens_cost": completion_tokens_cost
        }
        self.message_logger.info(json.dumps(log_entry, ensure_ascii=False))

    def _ensure_system_content(self):
        if self.system_content is None and getattr(self, 'run_mode', 'combined') != 'score_only':
            # tools_fmt = getattr(self, 'tools_format', "formatted")
            self.system_content = build_system_content(self.system_prompt, self._mcps, self.tools_format)
            print(f"[BUILD_SYSTEM] System content: {self.system_content}")


    def forward(self, **kwargs) -> dspy.Prediction:
        '''
        Forward pass for the MCP program. EVERYTHING IS BEING DONE IN HERE!!!
        '''
        # --- PROFILING ADDITIONS ---
        import time
        timings = {}
        data_sizes = {}
        t0 = time.perf_counter()
        # --- END PROFILING ADDITIONS ---

        unique_id = kwargs.get('id')
        question = kwargs.get('question')
        gt = kwargs.get('answer')
        tools_required = kwargs.get('tools_required')
        # print(f"tools_required: {tools_required}")

        # --- PROFILING ADDITIONS ---
        t1 = time.perf_counter()
        timings['init_vars'] = t1 - t0
        print(f"[PROFILE] init_vars took {timings['init_vars']:.4f}s")
        # --- END PROFILING ADDITIONS ---

        manager = ProcessManager()
        manager.lm_api_key = self.lm.api_key
        manager.lm_api_base = self.lm.api_base
        manager.model = self.lm.model
        manager.id = unique_id

        # --- PROFILING ADDITIONS ---
        t2 = time.perf_counter()
        timings['init_manager'] = t2 - t1
        print(f"[PROFILE] init_manager took {timings['init_manager']:.4f}s")
        # --- END PROFILING ADDITIONS ---

        self.run_logger.info(f"ID: {manager.id}, Starting forward pass for question: {question}")

        # The config is passed to the program instance by the EvaluateBench constructor.
        # We should use self.config instead of a global import.
        mcps = self.config['mcp_pool']

        self._ensure_system_content()
        messages = build_init_messages(self.system_prompt, mcps, question)
        system_prompt = messages[0][constants.CONTENT]

        # --- PROFILING ADDITIONS ---
        t3 = time.perf_counter()
        timings['build_init_messages'] = t3 - t2
        data_sizes['init_messages'] = len(str(messages))
        print(f"[PROFILE] build_init_messages took {timings['build_init_messages']:.4f}s")
        # --- END PROFILING ADDITIONS ---

        steps = 0
        all_completion_tokens = 0
        all_prompt_tokens = 0
        start_time = time.time()
        tools_called = []

        # --- PROFILING ADDITIONS ---
        loop_start = time.perf_counter()
        # --- END PROFILING ADDITIONS ---

        while not messages[-1][constants.ROLE] == constants.ASSISTANT and steps < self.max_steps:
            # --- PROFILING ADDITIONS ---
            step_start = time.perf_counter()
            # --- END PROFILING ADDITIONS ---
            response, completion_tokens, prompt_tokens = call_lm(messages, manager, self.run_logger, system_prompt=system_prompt)
            # --- PROFILING ADDITIONS ---
            step_end = time.perf_counter()
            print(f"[PROFILE] Step {steps}: call_lm took {step_end - step_start:.4f}s, response size: {len(str(response))}")
            # --- END PROFILING ADDITIONS ---

            all_completion_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            mcp_calls = response_parsing(response)
            # --- PROFILING ADDITIONS ---
            if hasattr(mcp_calls, 'mcps') and mcp_calls.mcps:
                print(f"[PROFILE] Step {steps}: response_parsing returned {len(mcp_calls.mcps)} calls")
            else:
                print(f"[PROFILE] Step {steps}: response_parsing returned 0 calls")
            # --- END PROFILING ADDITIONS ---

            if not mcp_calls.shutdown:
                for mcp_call in mcp_calls.mcps:
                    tools_called.append(mcp_call)
                    # print(f"Adding tool: {mcp_call}")

            # --- PROFILING ADDITIONS ---
            call_start = time.perf_counter()
            # --- END PROFILING ADDITIONS ---
            # --- Pass persistent client_cache to mcp_calling ---
            new_messages = mcp_calling(mcp_calls, manager, self.run_logger, self.config, client_cache=self.client_cache)
            # ---------------------------------------------------
            # --- PROFILING ADDITIONS ---
            call_end = time.perf_counter()
            print(f"[PROFILE] Step {steps}: mcp_calling took {call_end - call_start:.4f}s, returned {len(new_messages)} new messages")
            # --- END PROFILING ADDITIONS ---

            messages = build_messages(messages, new_messages)
            # --- PROFILING ADDITIONS ---
            print(f"[PROFILE] Step {steps}: build_messages, total messages: {len(messages)}")
            # --- END PROFILING ADDITIONS ---

            steps += 1

        # --- PROFILING ADDITIONS ---
        loop_end = time.perf_counter()
        timings['main_loop'] = loop_end - loop_start
        data_sizes['final_messages'] = len(str(messages))
        print(f"[PROFILE] main loop took {timings['main_loop']:.4f}s")
        # --- END PROFILING ADDITIONS ---

        end_time = time.time()

        # If the maximum number of steps is reached and there is still no answer
        if messages[-1][constants.ROLE] != constants.ASSISTANT:
            self.run_logger.warning("Maximum steps reached without getting an answer")
            messages.append({
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: "Maximum step limit exceeded, this problem cannot be solved",
            })

        self.run_logger.info(f"ID: {manager.id}, Forward pass completed successfully")
        prediction = messages[-1].get(constants.CONTENT, "")
        self.run_logger.info(f"ID: {manager.id}, prediction being passed to evaluation: {prediction[:50]}")

        # If we're only generating, skip evaluation and return a minimal prediction
        if getattr(self, 'run_mode', 'combined') == 'generate_only':
            self.log_messages(messages, question, None, (end_time - start_time), all_prompt_tokens,
                              all_completion_tokens)
            return dspy.Prediction(
                success="",
                question=question,
                ground_truth=gt,
                answer=messages[-1][constants.CONTENT],
                trace=messages,
                process_report=manager,
                evaluation_data=None,
                tool_calling_success="",
            )

        # If we're score-only, reuse saved answer and run just the evaluator
        if getattr(self, 'run_mode', 'combined') == 'score_only':
            saved_answer = self._get_saved_answer_for_id(unique_id)
            if not saved_answer:
                # If missing, fall back to a no-op minimal prediction
                self.log_messages(messages, question, None, (end_time - start_time), all_prompt_tokens,
                                  all_completion_tokens)
                return dspy.Prediction(
                    success="",
                    question=question,
                    ground_truth=gt,
                    answer="",
                    trace=messages,
                    process_report=manager,
                    evaluation_data=None,
                    tool_calling_success="",
                )
            # Evaluate saved answer without regenerating messages
            tools_called = []
            success, evaluation_data, tool_calling_success = self.evaluate_prediction(
                question, gt, tools_required, tools_called, saved_answer
            )
            self.log_messages(messages, question, success, (end_time - start_time), all_prompt_tokens,
                              all_completion_tokens)
            return dspy.Prediction(
                success=success,
                question=question,
                ground_truth=gt,
                answer=saved_answer,
                trace=messages,
                process_report=manager,
                evaluation_data=evaluation_data,
                tool_calling_success=tool_calling_success,
            )

        ## Everything till here is the same as the forward() in mcp_program.py

        ## Evaluation is done here!!!
        # --- PROFILING ADDITIONS ---
        eval_start = time.perf_counter()
        # --- END PROFILING ADDITIONS ---
        success, evaluation_data, tool_calling_success = self.evaluate_prediction(question, gt, tools_required, tools_called, messages[-1][constants.CONTENT])
        # --- PROFILING ADDITIONS ---
        eval_end = time.perf_counter()
        timings['evaluate_prediction'] = eval_end - eval_start
        if evaluation_data is not None:
            data_sizes['evaluation_data'] = len(str(evaluation_data))
        print(f"[PROFILE] evaluate_prediction took {timings['evaluate_prediction']:.4f}s")
        # --- END PROFILING ADDITIONS ---

        self.log_messages(messages, question, success, (end_time - start_time), all_prompt_tokens,
                          all_completion_tokens)
        self.run_logger.info(f"ID: {manager.id}, Evaluation completed successfully")

        # --- PROFILING ADDITIONS ---
        print(f"[PROFILE] Timings: {timings}")
        print(f"[PROFILE] Data sizes: {data_sizes}")
        # --- END PROFILING ADDITIONS ---

        return dspy.Prediction(
            success=success,
            question=question,
            ground_truth=gt,
            answer=messages[-1][constants.CONTENT],
            trace=messages,
            process_report=manager,
            evaluation_data=evaluation_data,
            tool_calling_success=tool_calling_success
        )