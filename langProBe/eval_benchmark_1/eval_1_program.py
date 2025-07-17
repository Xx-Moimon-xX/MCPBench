import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import List, Tuple, Optional
from langProBe.evaluation_utils import question_scorer

from langProBe.mcp_program import MCPPredict

import dspy
from openai import OpenAI

from langProBe.dspy_program import LangProBeDSPyMetaProgram
import langProBe.constants as constants

from langProBe.mcp_program import MCPPredict
from langProBe.program_utils import (
    call_lm,
    build_init_messages,
    build_messages,
    response_parsing,
    mcp_calling,
    ProcessManager
)

MCP_SAMPLE_SYSTEM_PROMPT = """
You are a helpful assistant. You are able to answer questions using different tools.  
The content of your available tools begins with ## Available Tools, indicating the collection of usable tools.  
Within the tool collection, each server is identified by ### server_name, where server_name represents the name of the server.  
Under each server, there are multiple tools (tool), and each tool starts with - tool_name, where tool_name is the name of the tool.  
The tool description includes:  
A brief text description outlining the functionality of the tool.  
Detailed information about input parameters, where each parameter includes: parameter name, parameter type, whether it is mandatory, and the purpose or description of the parameter.
If you have obtained the final result. Please provide your final answer enclosed within <answer></answer> tags. Ensure that only the final answer is included, without any additional explanations or commentary.
"""

class Eval1Predict(MCPPredict):
    def __init__(self, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="eval1"):
        super().__init__(max_steps, system_prompt, task_name)

    def evaluate_prediction(self, question: str, ground_truth: str, prediction: str) -> Tuple[bool, Optional[str]]:
        # This is mainly for gaia (not used anywhere else), probably not needed for eval1.
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.model.startswith("bedrock/"):
            answer_eval_manager.model = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        else:
            answer_eval_manager.model = "openai/deepseek-v3"

        return evaluate_final_answer(question, ground_truth, prediction, answer_eval_manager, self.run_logger)

        # return question_scorer(prediction, ground_truth, self.run_logger)

    def extract_last_answer(self, text):
        pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        matches = pattern.findall(text)

        if matches:
            return matches[-1]
        else:
            return None

    def forward(self, **kwargs) -> dspy.Prediction:
        '''
        This is the forward pass for the eval1 program.
        '''

        unique_id = kwargs.get('id')
        question = kwargs.get('question')
        gt = kwargs.get('answer')

        manager = ProcessManager()
        manager.lm_api_key = self.lm.api_key
        manager.lm_api_base = self.lm.api_base
        manager.model = self.lm.model
        manager.id = unique_id

        self.run_logger.info(f"ID: {manager.id}, Starting forward pass for question: {question}")

        from langProBe.evaluation import global_config
        mcps = global_config['mcp_pool']

        messages = build_init_messages(self.system_prompt, mcps, question)
        steps = 0
        all_completion_tokens = 0
        all_prompt_tokens = 0
        start_time = time.time()

        while not messages[-1][constants.ROLE] == constants.ASSISTANT and steps < self.max_steps:
            response, completion_tokens, prompt_tokens = call_lm(messages, manager, self.run_logger)
            all_completion_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            mcp_calls = response_parsing(response)

            new_messages = mcp_calling(mcp_calls, manager, self.run_logger)
            messages = build_messages(messages, new_messages)
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
        self.run_logger.info(f"ID: {manager.id}, prediction being passed to evaluation: {messages[-1][constants.CONTENT]}")

        ## Everything till here is the same as the forward() in mcp_program.py

        ## Evaluation is done here!!!
        success = self.evaluate_prediction(question, gt, self.extract_last_answer(messages[-1][constants.CONTENT]))
        self.log_messages(messages, question, success, (end_time - start_time), all_prompt_tokens,
                          all_completion_tokens)
        self.run_logger.info(f"ID: {manager.id}, Evaluation completed successfully")

        return dspy.Prediction(
            success=success,
            question=question,
            ground_truth=gt,
            answer=messages[-1][constants.CONTENT],
            trace=messages,
            process_report=manager
        )