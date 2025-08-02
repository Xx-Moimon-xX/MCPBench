import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import List, Tuple, Optional
# from langProBe.evaluation_utils import question_scorer, evaluate_final_answer_eval1

from langProBe.mcp_program import MCPPredict, MCPCall

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

EVAL_PROMPT_1 = """You are evaluating an LLM response against a prompt and expected answer. You have two evaluation tasks:

**Task 1: Prompt Adherence**
Evaluate if the response appropriately addresses the given prompt, including cases where the expected response indicates a failure.
- Score 5: Fully addresses all aspects of the prompt, including correct identification of failures if applicable
- Score 4: Addresses most aspects with minor gaps (may miss minor failure details)
- Score 3: Addresses some aspects but misses key elements or misrepresents failure cases
- Score 2: Minimally addresses the prompt or incorrectly describes failures
- Score 1: Fails to address the prompt meaningfully

**Task 2: Content Accuracy** 
Evaluate if the response conveys the same semantic meaning and core content as the expected response, including failure scenarios.
- Score 5: Conveys the same semantic meaning and captures all core concepts from the expected response, even if phrased differently
- Score 4: Conveys similar semantic meaning with most core concepts, minor differences in emphasis or detail
- Score 3: Conveys the general meaning but misses some important concepts or has notable semantic gaps
- Score 2: Partially aligns with expected meaning but has significant conceptual differences or omissions
- Score 1: Conveys different semantic meaning or contradicts the core concepts of the expected response

[BEGIN DATA]
[Prompt]: {prompt}
[Response]: {response}
[Expected Response]: {expected_response}
[END DATA]

For each task:
1. Analyze the relevant comparison
2. Assign a score (1-5) using the rubric above
3. Provide a yes/no answer (Prompt Adherence: "Does it address the prompt?" | Content Accuracy: "Does it convey the same semantic meaning?")
4. Give brief reasoning

Final score = Prompt Adherence + Content Accuracy (max 10)

Return as a JSON object with the following structure:
{{
  "prompt_adherence_score": <1-5>,
  "prompt_adherence_answer": "<yes/no>",
  "prompt_adherence_reasoning": "<brief explanation>",
  "content_accuracy_score": <1-5>,
  "content_accuracy_answer": "<yes/no>",
  "content_accuracy_reasoning": "<brief explanation>",
  "final_score": <2-10>
}}
"""

def eval_prompt_1_metric(example: dspy.Example, pred: dspy.Prediction):
    """
    Evaluates a prediction using the eval_prompt_1 evaluation prompt.
    Returns True if final_score >= 6, False otherwise.

    I DONT REALLY KNOW HOW TO USE THIS FUNCTION.
    """
    if not hasattr(pred, 'answer') or not pred.answer:
        return False
    
    prompt_text = EVAL_PROMPT_1.format(
        prompt=example.question,
        response=pred.answer,
        expected_response=example.answer
    )
    
    # Create a simple evaluation using the existing infrastructure
    # This is a simplified version - in practice you'd want to use the full ProcessManager
    try:
        # For now, we'll use a simple heuristic based on string similarity
        # In a real implementation, you'd call an LLM with the prompt_text
        
        # Simple fallback scoring based on string similarity
        response_lower = pred.answer.lower()
        expected_lower = example.answer.lower()
        
        # Basic similarity check
        if response_lower == expected_lower:
            return True
        elif any(word in response_lower for word in expected_lower.split()):
            return True
        else:
            return False
            
    except Exception as e:
        return False

def evaluate_final_answer_eval1(
            question: str, 
            ground_truth: str, 
            tools_required: List[str],
            tools_called: List[MCPCall],
            prediction: str, 
            manager: ProcessManager,
            logger: logging.Logger,
            ) -> Tuple[bool, Optional[str]]:
    prompt = EVAL_PROMPT_1.format(prompt=question, response=prediction, expected_response=ground_truth)
    messages = [
        {
            constants.ROLE: constants.USER,
            constants.CONTENT: prompt
        }
    ]
    logger.info(f"Starting evaluation of final answer with rubric")
    logger.info(f"question: {question}")
    logger.info(f"expected_response: {ground_truth[:50]}")
    logger.info(f"Prediction: {prediction[:50]}")
    response_content, _, _ = call_lm(messages, manager, logger, temperature=0.01)
    
    json_str = ""
    try:
        # Try to extract JSON from markdown code block if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for raw JSON possibly with leading/trailing text
            json_start = response_content.find('{')
            json_end = response_content.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end+1]
            else:
                raise json.JSONDecodeError("No JSON object found in response", response_content, 0)

        scores_data = json.loads(json_str)
        # print(f"DEBUG: scores_data: {scores_data}")
        
        prompt_adherence_score = scores_data.get("prompt_adherence_score")
        content_accuracy_score = scores_data.get("content_accuracy_score")
        final_score = scores_data.get("final_score")

        logger.info(f"Extracted scores: Prompt Adherence={prompt_adherence_score}, Content Accuracy={content_accuracy_score}, Final={final_score}")

        tool_calling_success = True
        ## Checking if the required tools were called.
        if tools_called:
            called_tool_names = [call.mcp_tool_name for call in tools_called]
            for tool in tools_required:
                if tool not in called_tool_names:
                    # print(f"Tool {tool} was not called.")
                    tool_calling_success = False
                    # return False, None, False
        else:
            tool_calling_success = False

        if final_score is None:
            logger.error("Could not find 'final_score' in the LLM response.")
            return False, "Missing 'final_score' in response", tool_calling_success

        # Success is defined as final_score >= 6 (based on eval_prompt_1_metric docstring)
        is_success = int(final_score) >= 6 and tool_calling_success
        
        return is_success, json.dumps(scores_data), tool_calling_success

    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON from LLM response: {response_content}"
        logger.error(error_msg)
        return False, error_msg
    except (KeyError, TypeError) as e:
        error_msg = f"Error accessing scores from parsed JSON: {e}. Data: {json_str}"
        logger.error(error_msg)
        return False, error_msg

    
class Eval1Predict(MCPPredict):
    '''
    Program that is run to get responses. Called Eval1Predict and it is a child class of MCPPredict.
    '''
    def __init__(self, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="eval1"):
        super().__init__(max_steps, system_prompt, task_name)

    
    def evaluate_prediction(self, question: str, ground_truth: str, tools_required: List[str], tools_called: List[MCPCall], prediction: str) -> Tuple[bool, Optional[str]]:
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.eval_model:
            answer_eval_manager.model = self.lm.eval_model
        else:
            answer_eval_manager.model = self.lm.model

        return evaluate_final_answer_eval1(question, ground_truth, tools_required, tools_called, prediction, answer_eval_manager, self.run_logger)
        
        # TO DO: ID here is wrong, i.e. answer_eval_manager isn't correct or whatever. Need to check/fix this.
        # self.run_logger.info(f"ID: {answer_eval_manager.id}, Evaluation completed successfully")
        # self.run_logger.info(f"ID: {answer_eval_manager.id}, scores_data: {scores_data}")
        # return is_success, evaluation_data  

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
        tools_required = kwargs.get('tools_required')
        # print(f"tools_required: {tools_required}")

        manager = ProcessManager()
        manager.lm_api_key = self.lm.api_key
        manager.lm_api_base = self.lm.api_base
        manager.model = self.lm.model
        manager.id = unique_id

        self.run_logger.info(f"ID: {manager.id}, Starting forward pass for question: {question}")

        # The config is passed to the program instance by the EvaluateBench constructor.
        # We should use self.config instead of a global import.
        mcps = self.config['mcp_pool']
        
        messages = build_init_messages(self.system_prompt, mcps, question)
        self.run_logger.debug(f"ID: {manager.id}, Build initial messages: {messages}")
        steps = 0
        all_completion_tokens = 0
        all_prompt_tokens = 0
        start_time = time.time()
        tools_called = []

        # Debug: Print messages before calling call_lm
        # print(f"[DEBUG] eval_1_program.forward: messages before call_lm:")
        # for i, m in enumerate(messages):
        #     print(f"  Message {i}: role={m.get('role')} content={repr(m.get('content'))}")
        #     if not m.get('content'):
        #         print(f"  [WARNING] Message {i} has empty or missing content!")
        while not messages[-1][constants.ROLE] == constants.ASSISTANT and steps < self.max_steps:
            response, completion_tokens, prompt_tokens = call_lm(messages, manager, self.run_logger)
            self.run_logger.debug(f"ID: {manager.id}, Response from LLM: {response}")

            all_completion_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            mcp_calls = response_parsing(response)

            if not mcp_calls.shutdown:
                for mcp_call in mcp_calls.mcps:
                    tools_called.append(mcp_call)
                    # print(f"Adding tool: {mcp_call}")

            self.run_logger.debug(f"ID: {manager.id}, After response parsing: {mcp_calls}")

            new_messages = mcp_calling(mcp_calls, manager, self.run_logger, self.config)
            messages = build_messages(messages, new_messages)
            steps += 1

        end_time = time.time()
        # print(f"Tools called: {tools_called}")

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

        ## Everything till here is the same as the forward() in mcp_program.py

        ## Evaluation is done here!!!

        success, evaluation_data, tool_calling_success = self.evaluate_prediction(question, gt, tools_required, tools_called, messages[-1][constants.CONTENT])
        self.log_messages(messages, question, success, (end_time - start_time), all_prompt_tokens,
                          all_completion_tokens)
        self.run_logger.info(f"ID: {manager.id}, Evaluation completed successfully")

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