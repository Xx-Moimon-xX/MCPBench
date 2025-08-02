import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import List, Tuple, Optional
from langProBe.benchmark import Benchmark
from langProBe.config_utils import read_jsonl
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

EVAL_PROMPT_2 = """You are an expert evaluator assessing how well an LLM response matches expected responses.

**EVALUATION DATA:**
[BEGIN DATA]
[Prompt]: {prompt}
[LLM Response]: {response}
[Expected Response 1]: {expected_response_1}
[Expected Response 2]: {expected_response_2}
[Expected Response 3]: {expected_response_3}
[END DATA]

**TASK:**
1. Compare the LLM response to each expected response and identify which ONE it most closely matches
2. Score the match quality using this 5-point scale:
   - **5 (Excellent):** Same core meaning, even if worded differently
   - **4 (Good):** Minor differences only (slight wording variations)
   - **3 (Partial):** Significant differences affecting clarity/completeness
   - **2 (Poor):** Some relation but fails to convey correct meaning
   - **1 (No Match):** No meaningful match in meaning/content/intent
3. Determine acceptance: scores 3-5 = "yes", scores 1-2 = "no"

**CRITICAL OUTPUT REQUIREMENTS:**
- You MUST return ONLY a JSON object with EXACTLY these 4 fields
- Use EXACTLY these field names (case-sensitive): "selected_expected_response", "score", "answer", "reasoning"
- DO NOT add any other fields or modify field names
- DO NOT include any text before or after the JSON
- DO NOT include markdown code blocks or formatting
- DO NOT include a comma after the last field in the JSON object.
- The value of the "reasoning" field should be a string, with no extra characters (such as commas, periods, or whitespace) after the closing quotation mark.

**REQUIRED JSON FORMAT:**
{{
    "selected_expected_response": "<exact copy of the expected response you selected>",
    "score": <integer from 1 to 5>,
    "answer": "<exactly 'yes' or 'no' in lowercase>",
    "reasoning": "<2-3 sentence explanation comparing LLM response to selected expected response>"
}}

Focus on semantic meaning over exact wording. When uncertain between scores, choose the lower score.

Return only the JSON object now:"""


class MCPBench2(Benchmark):
    '''
    Concrete benchmark for MCP tasks. Loads test data from a JSONL file or provided data, and creates dspy.Example objects.
    '''
    def __init__(self, dataset_mode="lite", dataset_path=None, missing_data=[]):
        '''
        Initializes MCPBench with a dataset mode, path, and optional missing data.
        '''
        self.dataset_path = dataset_path
        self.missing_data = missing_data
        super().__init__(dataset_mode=dataset_mode)

    def init_dataset(self):
        '''
        Loads the dataset and test set from the given path or missing_data, and creates dspy.Example objects for each entry.
        '''
        self.dataset = []
        self.test_set = []
        if self.missing_data:
            test_raw_data = self.missing_data
        else:
            test_raw_data = read_jsonl(self.dataset_path)
        
        for test_data in test_raw_data:
            self.test_set.append(
                dspy.Example(
                    id=test_data["unique_id"],
                    question=test_data["Prompt"],
                    answer1=test_data["Answer1"],
                    answer2=test_data["Answer2"],
                    answer3=test_data["Answer3"],
                    tools_required=test_data["tools_required"]
                ).with_inputs("id", "question", "answer1", "answer2", "answer3", "tools_required", "config")
            )

def evaluate_final_answer_eval2(
            question: str, 
            ground_truth_1: str, 
            ground_truth_2: str, 
            ground_truth_3: str, 
            tools_required: List[str],
            tools_called: List[MCPCall],
            prediction: str, 
            manager: ProcessManager,
            logger: logging.Logger,
            ) -> Tuple[bool, Optional[str]]:
    prompt = EVAL_PROMPT_2.format(prompt=question, response=prediction, expected_response_1=ground_truth_1, expected_response_2=ground_truth_2, expected_response_3=ground_truth_3)
    messages = [
        {
            constants.ROLE: constants.USER,
            constants.CONTENT: prompt
        }
    ]
    logger.info(f"Starting evaluation of final answer with rubric")
    logger.info(f"question: {question}")
    logger.info(f"expected_response 1: {ground_truth_1[:50]}")
    logger.info(f"expected_response 2: {ground_truth_2[:50]}")
    logger.info(f"expected_response 3: {ground_truth_3[:50]}")
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
        
        score = scores_data.get("score")
        answer = scores_data.get("answer")
        reasoning = scores_data.get("reasoning")
        selected_expected_response = scores_data.get("selected_expected_response")

        logger.info(f"Extracted score: {score}, answer: {answer}, selected_expected_response: {selected_expected_response}, reasoning: {reasoning}")

        tool_calling_success = True
        ## Checking if the required tools were called.
        if tools_called:
            called_tool_names = [call.mcp_tool_name for call in tools_called]
            for tool in tools_required:
                if tool not in called_tool_names:
                    print(f"Tool {tool} was not called.")
                    tool_calling_success = False
                    # return False, None, False
        else:
            tool_calling_success = False

        if score is None or answer is None:
            logger.error("Could not find 'score' or 'answer' in the LLM response.")
            return False, "Missing 'score' or 'answer' in response", tool_calling_success

        # Success is defined by the 'answer' field being 'yes' and tool calling success
        is_success = answer == "yes" and tool_calling_success
        
        return is_success, json.dumps(scores_data), tool_calling_success

    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON from LLM response: {response_content}"
        logger.error(error_msg)
        return False, error_msg
    except (KeyError, TypeError) as e:
        error_msg = f"Error accessing scores from parsed JSON: {e}. Data: {json_str}"
        logger.error(error_msg)
        return False, error_msg

    
class Eval2Predict(MCPPredict):
    '''
    Program that is run to get responses. Called Eval1Predict and it is a child class of MCPPredict.
    '''
    def __init__(self, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="eval1"):
        super().__init__(max_steps, system_prompt, task_name)

    
    def evaluate_prediction(self, question: str, ground_truth_1: str, ground_truth_2: str, ground_truth_3: str, tools_required: List[str], tools_called: List[MCPCall], prediction: str) -> Tuple[bool, Optional[str]]:
        # This is mainly for gaia (not used anywhere else), probably not needed for eval1.
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.eval_model:
            answer_eval_manager.model = self.lm.eval_model
        else:
            answer_eval_manager.model = self.lm.model

        return evaluate_final_answer_eval2(question, ground_truth_1, ground_truth_2, ground_truth_3, tools_required, tools_called, prediction, answer_eval_manager, self.run_logger)
        
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
        gt1 = kwargs.get('answer1')
        gt2 = kwargs.get('answer2')
        gt3 = kwargs.get('answer3')
        tools_required = kwargs.get('tools_required')
        print(f"tools_required: {tools_required}")

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
        steps = 0
        all_completion_tokens = 0
        all_prompt_tokens = 0
        start_time = time.time()
        tools_called = []

        while not messages[-1][constants.ROLE] == constants.ASSISTANT and steps < self.max_steps:
            response, completion_tokens, prompt_tokens = call_lm(messages, manager, self.run_logger)
            all_completion_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            mcp_calls = response_parsing(response)

            if not mcp_calls.shutdown:
                for mcp_call in mcp_calls.mcps:
                    tools_called.append(mcp_call)
                    print(f"Adding tool: {mcp_call}")
            
            self.run_logger.debug(f"ID: {manager.id}, After response parsing: {mcp_calls}")

            new_messages = mcp_calling(mcp_calls, manager, self.run_logger, self.config)
            messages = build_messages(messages, new_messages)
            steps += 1

        end_time = time.time()
        print(f"Tools called: {tools_called}")

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

        success, evaluation_data, tool_calling_success = self.evaluate_prediction(question, gt1, gt2, gt3, tools_required, tools_called, messages[-1][constants.CONTENT])
        self.log_messages(messages, question, success, (end_time - start_time), all_prompt_tokens,
                          all_completion_tokens)
        # Get the selected expected response from the evaluation data
        selected_expected_response = json.loads(evaluation_data).get("selected_expected_response")


        self.run_logger.info(f"ID: {manager.id}, Evaluation completed successfully")

        return dspy.Prediction(
            success=success,
            question=question,
            ground_truth=selected_expected_response,
            answer=messages[-1][constants.CONTENT],
            trace=messages,
            process_report=manager,
            evaluation_data=evaluation_data, 
            tool_calling_success=tool_calling_success
        )