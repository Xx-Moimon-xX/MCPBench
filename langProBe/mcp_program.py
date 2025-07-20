import dspy
from pydantic import BaseModel, Field
from langProBe.program_utils import (
    call_lm, 
    build_init_messages, 
    build_messages,
    response_parsing,
    mcp_calling,
    ProcessManager
)
import time
from langProBe.evaluation_utils import evaluate_final_answer
import langProBe.constants as constants
import logging
import os
from datetime import datetime
import json
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

class LangProBeMCPMetaProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = MCP_LM()
    def setup_lm(self, lm, api_key=None, api_base=None):
        self.lm.model = lm
        self.lm.api_key = api_key
        self.lm.api_base = api_base

    def program_type(self):
        return "mcp"
    

class MCPPredict(LangProBeMCPMetaProgram, dspy.Module):
    '''
    This is the program/system that is run to get responses. Called MCPPredict.
    '''
    def __init__(self, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="mcp_sample", config=None):
        super().__init__()
        self.system_prompt = system_prompt
        self.task_name = task_name
        self.max_steps = max_steps
        self.max_length = 30000
        self.config = config
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

    # def set_dataset(self, dataset):
    #     self.dataset = dataset
        # self.setup_loggers()

    def set_log_path(self, path):
        self.log_path = path
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

        # Set up message log
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
        
        message_log_file = f'{new_log_dir}/{self.task_name}_messages_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        message_handler = logging.FileHandler(message_log_file, encoding='utf-8')
        self.message_logger.addHandler(message_handler)

    def evaluate_prediction(self, question: str, ground_truth: str, prediction: str) -> Tuple[bool, Optional[str]]:
        '''
        Main function to evaluate the prediction.
        '''
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.model.startswith("bedrock/"):
            answer_eval_manager.model = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        else:
            answer_eval_manager.model = "openai/deepseek-v3"
        # # Set AWS credentials if available
        # answer_eval_manager.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        # answer_eval_manager.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        # answer_eval_manager.aws_region = os.getenv("AWS_REGION")
        return evaluate_final_answer(question, ground_truth, prediction, answer_eval_manager, self.run_logger)

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


    def forward(self, **kwargs) -> dspy.Prediction:
        '''
        Forward pass for the MCP program. EVERYTHING IS BEING DONE IN HERE!!!
        '''
        unique_id = kwargs.get('id')
        question = kwargs.get('question')
        gt = kwargs.get('answer')

        manager = ProcessManager()
        manager.lm_api_key = self.lm.api_key
        manager.lm_api_base = self.lm.api_base
        manager.model = self.lm.model
        manager.id = unique_id

        # # Set AWS credentials if available
        # manager.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        # manager.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        # manager.aws_region = os.getenv("AWS_REGION", "us-east-1")

        self.run_logger.info(f"ID: {manager.id}, Starting forward pass for question: {question}")


        # Use config passed to the instance
        mcps = self.config['mcp_pool']


        messages = build_init_messages(self.system_prompt, mcps, question)
        steps = 0
        all_completion_tokens = 0
        all_prompt_tokens = 0
        start_time = time.time()

        # Iterative processing loop until the last message is an assistant message or the maximum number of steps is reached
        while not messages[-1][constants.ROLE] == constants.ASSISTANT and steps < self.max_steps:
            response, completion_tokens, prompt_tokens= call_lm(messages, manager, self.run_logger)
            all_completion_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            mcp_calls = response_parsing(response)

            new_messages = mcp_calling(mcp_calls, manager, self.run_logger, self.config)

            self.run_logger.info(f"New messages from MCP calling: {new_messages}")
            messages = build_messages(messages, new_messages)
            self.run_logger.info(f"Messages concatenated with new messages: {messages}")
            steps += 1

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

        # IMPORTANT: This is where the evaluation is done !!!!!!
        success = self.evaluate_prediction(question, gt, messages[-1][constants.CONTENT])
        self.log_messages(messages, question, success, (end_time-start_time), all_prompt_tokens, all_completion_tokens)
        self.run_logger.info(f"ID: {manager.id}, Evaluation completed successfully")
        # self.run_logger.info("==" * 50)

        return dspy.Prediction(
            success=success,
            question=question,
            ground_truth=gt,
            answer=messages[-1][constants.CONTENT],
            trace=messages,
            process_report=manager
        )