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

EVAL_PROMPT_3 = """
You are evaluating an LLM response against a prompt using multiple weighted criteria. Each criterion has its own evaluation scale and weight.

**Evaluation Criteria:**

{criteria_definitions}

**Scoring Instructions:**
- Evaluate each criterion independently using its specific scale
- Consider the weight of each criterion in your assessment
- Provide clear reasoning for each score

[BEGIN DATA]
[Prompt]: {prompt}
[Response]: {response}
[Expected Response]: {expected_response}
[END DATA]

For each criterion:
1. Analyze the response against the criterion description
2. Assign a score using the criterion's evaluation scale
3. Provide brief reasoning for your score

Return as a JSON object with the following structure, with no additional text or commentary:
{{
  "criterion_scores": {{
    {criterion_json_structure}
  }}
}}
"""

def generate_evaluation_prompt(prompt, response, expected_response, rubric_data):
    criteria_definitions = []
    criterion_json_fields = []
    
    for criterion in rubric_data:
        # Build criterion definition
        criterion_def = f"""**Criterion: {criterion['criterion']} (Weight: {criterion['weight']})**
Description: {criterion['description']}
Evaluation Scale:"""
        
        for scale_item in criterion['evaluation_scale']:
            criterion_def += f"\n- Score {scale_item['score']}: {scale_item['condition']}"
        
        criteria_definitions.append(criterion_def)
        
        # Build JSON field
        field_name = criterion['criterion'].lower().replace(' ', '_')
        json_field = f'''"{field_name}": {{
            "score": <1-5>,
            "reasoning": "<brief explanation>",
            "weight": {criterion['weight']}
        }}'''
        criterion_json_fields.append(json_field)
    
    # criterion_json_fields.append("")
    
    # Combine into full prompt
    full_prompt = EVAL_PROMPT_3.format(
        prompt=prompt,
        response=response,
        expected_response=expected_response,
        criteria_definitions='\n\n'.join(criteria_definitions),
        criterion_json_structure=',\n    '.join(criterion_json_fields)
    )
    
    return full_prompt

class MCPBench3(Benchmark):
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
                    answer=test_data["Answer"],
                    rubric_data=test_data["Rubrics"],
                    tools_required=test_data["tools_required"]
                ).with_inputs("id", "question", "answer", "rubric_data", "tools_required", "config")
            )

def evaluate_final_answer_eval3(
            question: str, 
            ground_truth: str, 
            rubric_data: dict,
            tools_required: List[str],
            tools_called: List[MCPCall],
            prediction: str, 
            manager: ProcessManager,
            logger: logging.Logger,
            ) -> Tuple[bool, Optional[str]]:

    prompt = generate_evaluation_prompt(prompt=question, response=prediction, expected_response=ground_truth, rubric_data=rubric_data)
    # prompt = EVAL_PROMPT_3.format(prompt=question, response=prediction, expected_response_1=ground_truth_1, expected_response_2=ground_truth_2, expected_response_3=ground_truth_3)
    print(f"Eval 3 prompt: {prompt}")
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
        
        criterion_scores = scores_data.get("criterion_scores", {})
        final_weighted_score = 0
        max_possible_score = 5 * len(rubric_data)
        
        for criterion in criterion_scores.values():
            score = criterion["score"]
            weight = criterion["weight"]
            reasoning = criterion["reasoning"]
            
            if score is not None and weight is not None:
                final_weighted_score += score * weight
            
            # Find the max score for this criterion from rubric_data
            # for rubric_item in rubric_data:
            #     if rubric_item['criterion'].lower().replace(' ', '_') == criterion_name:
            #         max_score_for_criterion = max(s['score'] for s in rubric_item.get('evaluation_scale', []))
            #         max_possible_score += max_score_for_criterion * 1
            #         break
        
        if max_possible_score > 0:
            final_score = final_weighted_score / max_possible_score
        else:
            final_score = 0

        logger.info(f"Final weighted score: {final_weighted_score}, Max possible score: {max_possible_score}, Final score: {final_score}")

        scores_data['final_weighted_score'] = final_weighted_score
        scores_data['max_possible_score'] = max_possible_score
        scores_data['final_score'] = final_score

        print(f"scores_data eval_3: {scores_data}")

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

        # For now, let's consider a success if the final score is above a certain threshold, e.g., 0.5
        # and all required tools were called.
        # You might want to adjust this logic based on your specific needs.
        is_success = (final_score >= 0.5) and tool_calling_success
        
        return is_success, json.dumps(scores_data), tool_calling_success

    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON from LLM response: {response_content}"
        logger.error(error_msg)
        return False, error_msg
    except (KeyError, TypeError) as e:
        error_msg = f"Error accessing scores from parsed JSON: {e}. Data: {json_str}"
        logger.error(error_msg)
        return False, error_msg

    
class Eval3Predict(MCPPredict):
    '''
    Program that is run to get responses. Called Eval1Predict and it is a child class of MCPPredict.
    '''
    def __init__(self, max_steps=5, system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, task_name="eval3"):
        super().__init__(max_steps, system_prompt, task_name)

    
    def evaluate_prediction(self, question: str, ground_truth: str, rubric_data: dict, tools_required: List[str], tools_called: List[MCPCall], prediction: str) -> Tuple[bool, Optional[str]]:
        # This is mainly for gaia (not used anywhere else), probably not needed for eval1.
        answer_eval_manager = ProcessManager()
        answer_eval_manager.lm_api_key = self.lm.api_key
        answer_eval_manager.lm_api_base = self.lm.api_base
        # Use the same model type as the main LM for evaluation
        if self.lm.model.startswith("bedrock/"):
            answer_eval_manager.model = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        else:
            answer_eval_manager.model = "openai/deepseek-v3"

        return evaluate_final_answer_eval3(question, ground_truth, rubric_data, tools_required, tools_called, prediction, answer_eval_manager, self.run_logger)
        
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
        rubric_data = kwargs.get('rubric_data')
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

        success, evaluation_data, tool_calling_success = self.evaluate_prediction(question, gt, rubric_data, tools_required, tools_called, messages[-1][constants.CONTENT])
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