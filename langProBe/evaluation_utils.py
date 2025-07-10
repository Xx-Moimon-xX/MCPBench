import json
import dspy
from typing import List, Tuple, Optional
from langProBe.program_utils import call_lm, ProcessManager
import langProBe.constants as constants
import logging
import re
import string
import warnings
import os
import logging
import numpy as np


EVALUATE_PROMPT = """For the following question: {question}

Please determine whether the predicted answer is correct. Consider it correct if it addresses the key information:

Predicted answer: {prediction}
Correct answer: {ground_truth}

Return only True or False."""

def evaluate_final_answer(
            question: str, 
            ground_truth: str, 
            prediction: str, 
            manager: ProcessManager,
            logger: logging.Logger,
            ) -> Tuple[bool, Optional[str]]:
    prompt = EVALUATE_PROMPT.format(question=question, prediction=prediction, ground_truth=ground_truth)
    messages = [
        {
            constants.ROLE: constants.USER,
            constants.CONTENT: prompt
        }
    ]
    logger.info(f"开始评测final answer")
    logger.info(f"question: {question}")
    logger.info(f"ground_truth: {ground_truth}")
    logger.info(f"prediction: {prediction}")
    response_content, _, _ = call_lm(messages, manager, logger, temperature=0.01)
    return "true" in response_content.lower()


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
        s: str,
        char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)

def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def question_scorer(
        model_answer: str,
        ground_truth: str,
        logger: logging.Logger
) -> Tuple[bool, Optional[str]]:
    def is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if model_answer is None:
        model_answer = "None"
        logger.debug("Model answer is None. Converted to string 'None'.")

    # If ground truth is a number
    if is_float(ground_truth):
        info = f"Evaluating '{model_answer}' as a number."
        logger.info(info)
        normalized_answer = normalize_number_str(model_answer)
        try:
            result = normalized_answer == float(ground_truth)
            logger.debug(f"Normalized model answer: {normalized_answer}, Ground truth: {ground_truth}, Result: {result}")
            return result
        except ValueError as e:
            error_msg = f"Normalization error: {e}"
            logger.error(error_msg)
            return False

    # If ground truth is a list
    elif any(char in ground_truth for char in [",", ";"]):
        info = f"Evaluating '{model_answer}' as a comma/semi-colon separated list."
        logger.info(info)

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        logger.debug(f"Ground truth elements: {gt_elems}")
        logger.debug(f"Model answer elements: {ma_elems}")

        # Check if lengths are the same
        if len(gt_elems) != len(ma_elems):
            warning_msg = "Answer lists have different lengths."
            logger.warning(warning_msg)
            return False

        # Compare each element as float or string
        comparisons = []
        for idx, (ma_elem, gt_elem) in enumerate(zip(ma_elems, gt_elems), start=1):
            if is_float(gt_elem):
                try:
                    normalized_ma_elem = normalize_number_str(ma_elem)
                    comparison = normalized_ma_elem == float(gt_elem)
                    logger.debug(f"Element {idx}: Normalized model answer element '{normalized_ma_elem}' == Ground truth element '{float(gt_elem)}': {comparison}")
                except ValueError as e:
                    error_msg = f"Normalization error at element {idx}: {e}"
                    logger.error(error_msg)
                    return False
            else:
                normalized_ma = normalize_str(ma_elem, remove_punct=False)
                normalized_gt = normalize_str(gt_elem, remove_punct=False)
                comparison = normalized_ma == normalized_gt
                logger.debug(f"Element {idx}: Normalized model answer element '{normalized_ma}' == Ground truth element '{normalized_gt}': {comparison}")
            comparisons.append(comparison)

        all_correct = all(comparisons)
        if not all_correct:
            detail_msg = "Mismatch found in list elements."
            logger.info(detail_msg)
            return all_correct
        logger.debug("All list elements match the ground truth.")
        return all_correct

    # If ground truth is a string
    else:
        info = f"Evaluating '{model_answer}' as a string."
        logger.info(info)
        normalized_ma = normalize_str(model_answer)
        normalized_gt = normalize_str(ground_truth)
        result = normalized_ma == normalized_gt
        logger.debug(f"Normalized model answer: '{normalized_ma}' == Normalized ground truth: '{normalized_gt}': {result}")
        return result

def mcp_metric(example: dspy.Example, pred: dspy.Prediction):
    return pred.success



def extract_questions(data, key):
    """从数据中提取指定字段（如 Prompt 或 question）用于比较"""
    questions = set()
    for item in data:
        questions.add(item[key])
    return questions

def find_missing_entries(data_a, data_b):
    # data_a是原数据，data_b是已经跑了的数据
 
    questions_in_b = extract_questions(data_b, 'question')

    # 找出在B中不存在的A条目
    missing_entries = [item for item in data_a if item['Prompt'] not in questions_in_b]

    return missing_entries

import logging

import os
import logging

def replace_logger_filehandler(new_log_name):
    """
    替换 logger 中已有的 FileHandler，并为每个 logger 保留其原有的 formatter。
    同时删除原有日志文件。

    :param new_log_name: 新的日志文件名（不带后缀）
    """

    def update_handler(logger, file_suffix):
        old_log_paths = []
        formatter = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if formatter is None:
                    formatter = handler.formatter
                old_log_paths.append(handler.baseFilename)

        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)

        for log_path in old_log_paths:
            if os.path.exists(log_path):
                try:
                    os.remove(log_path)
                except Exception as e:
                    pass

        if logger.name == 'MCPPredictRunLogger':
            new_name = new_log_name.replace("message", "run")
        else:
            new_name = new_log_name 

        new_handler = logging.FileHandler(f"{new_name}.{file_suffix}", mode='a', encoding='utf-8')
        if formatter:
            new_handler.setFormatter(formatter)
        logger.addHandler(new_handler)

    run_logger = logging.getLogger('MCPPredictRunLogger')
    update_handler(run_logger, 'log')

    message_logger = logging.getLogger('MCPPredictMessageLogger')
    update_handler(message_logger, 'jsonl')



if __name__ == "__main__":
    print(question_scorer("123", "123"))
    