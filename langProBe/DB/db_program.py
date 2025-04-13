import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import List, Tuple, Optional

import dashscope
import dspy
from openai import OpenAI

from langProBe.dspy_program import LangProBeDSPyMetaProgram


class ResponseManager:
    """Manages the processing of responses and web search operations."""

    def __init__(self):
        self.client = None
        self.max_length = 5000

    def initialize_client(self):
        from langProBe.evaluation import global_config
        """延迟初始化客户端，避免在导入时启动进程"""
        if self.client is None:
            try:
                from .DB_utils.synced_mcp_client import SyncedMcpClient
                if 'url' in global_config.keys():
                    url = global_config['url']
                else:
                    url = "http://localhost:" + str(global_config['port']) + '/sse'
                self.client = SyncedMcpClient(server_url=url)

                print("Client initialized")
                self.client.list_tools()
            except Exception as e:
                print(f"Failed to initialize search client: {str(e)}")
                # 设置一个默认值，避免多次尝试初始化失败的客户端
                self.client = None

    def extract_search_query(self, response: str) -> Optional[str]:
        """Extract search query from response if it contains WebSearch tags.

        Args:
            response: The response string to process

        Returns:
            The search query if found, None otherwise
        """
        search_match = re.search(r'<DB>(.*?)</DB>', response)
        return search_match.group(1) if search_match else None

    def perform_db(self, query: str) -> str:
        """Perform web search using the provided query."""
        from langProBe.evaluation import global_config
        try:
            self.initialize_client()
            if self.client is None:
                return f"Error: Search client could not be initialized. Using default response for '{query}'."

            # result = self.client.call_tool("fetch", {"url": "https://baike.baidu.com/item/唐纳德·特朗普/9916449"})
            # result = self.client.call_tool("GoogleInternetSearch", {"keyword": query})
            result = self.client.call_tool(global_config['tool_name'], {global_config['tool_keyword']: query})

            texts = [item.text for item in result.content]
            result_str = ''.join(texts)
            return result_str[:self.max_length]
        except Exception as e:
            print(f"db error: {str(e)}")
            print(traceback.format_exc())
            # 返回一个默认结果，避免程序崩溃
            return f"Error performing db for '{query}': {str(e)}"



class DBPredictSignature(dspy.Signature):
    """Handles questions that may require db.

    Input contains:
    1. The question that needs to be answered
    2. Past search steps and their results

    Output can be either:
    1. If more search is needed: output in format <DB>search_query</DB>
    2. If question can be answered: direct answer

    Example:
    Input question: "Who is the current President of the United States?"
    - If no search has been done: <DB>current President of United States</DB>
    - If sufficient information exists: "Joe Biden is the current President of the United States"
    """

    question = dspy.InputField(description="Question to be answered", format=str)
    past_steps = dspy.InputField(description="Past search steps and results", format=str)
    response = dspy.OutputField(description="Answer or new search request (format: <DB>query</DB>)",
                                format=str)


class DBPredict(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, max_steps=5, module=None, log_file='logs/db_dummy.log'):
        super().__init__()
        if module is None:
            self.module = dspy.Predict(signature=DBPredictSignature)
        else:
            self.module = module
        self.max_steps = max_steps
        self.response_manager = ResponseManager()
        self.max_length = 30000

        # 配置日志记录器
        self.logger = logging.getLogger('DBPredictLogger')
        self.logger.setLevel(logging.INFO)
        # 防止重复添加处理器
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    def add_logger(self ,name):
        log_file = 'logs/db_' + name.replace(' ' ,'_' )+ '.log'
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def format_trace(self, trace: List[Tuple[str, str]]) -> str:
        if not trace:
            return ""

        full_trace = "\n\n".join([
            f"Search Query: {query}\nResult: {result}"
            for query, result in trace
        ])

        if len(full_trace) > self.max_length:
            return full_trace[:self.max_length]

        return full_trace

    def evaluate_prediction(self, prediction: str, question: str) -> Tuple[bool, Optional[str]]:
        # TODO:这块后面还要改，现在先这样
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'car_bi.jsonl')
        result = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                q = data.get('Prompt')
                a = data.get('Answer')
                if q and a:
                    result[q] = a

        ground_truth = result.get(question)

        prompt = f"""对于以下问题：{question}

请判断预测答案是否回答正确，回答对关键信息就算正确:

预测答案: {prediction}
正确答案: {ground_truth}

只需要返回True或False。
"""
        messages = [{"role": "user", "content": prompt}]
        model_name = "deepseek-v3"
        try:
            dashscope.api_key = key=os.environ.get('MODEL_KEY')
            response = dashscope.Generation.call(
                model=model_name,
                messages=messages,
                result_format="message",
                temperature=0.01
            )
            response_content = response['output']['choices'][0]['message']['content']
        except Exception as e:
            print(f"dash error: {str(e)}")
            return False, ground_truth

        return response_content.lower() == "true", ground_truth

    def call_lm(self, query: str, trace: List) -> tuple[str | None, int, int]:
        from langProBe.evaluation import global_config
        schema=''
        with open(os.path.join(os.path.dirname(__file__),'data','schema.txt'), 'r', encoding='utf-8') as f:
            schema='\n'.join(f.readlines())

        key=os.environ.get('MODEL_KEY')
        endpoint = os.environ.get('MODEL_ENDPOINT',"https://dashscope.aliyuncs.com/compatible-mode/v1")
        oai = OpenAI(api_key=key,
                     base_url=endpoint)

        SYSTEM_PROMPT_SQL = f"""
        Handles questions that may require fetch data from database.

        Here is the database schema
        {schema}
                
        Input contains:
        1. The question that needs to be answered
        2. Past search steps and their results

        Output can be either:
        1. If more search is needed: output in format <DB>SQL</DB>
        2. If question can be answered: direct answer

        Example:
        Input question: "华东区域的系列A总库存是多少?"
        - If no search has been done: <DB>select sum(quantity) from inventory where region = '华东' and car_series='系列A';</DB>
        - If sufficient information exists: "374"
        """

        SYSTEM_PROMPT_NL = f"""
        Handles questions that may require fetch data from database.

        Input contains:
        1. The question that needs to be answered
        2. Past search steps and their results

        Output can be either:
        1. If more search is needed: output in format <DB>search_query</DB>
        2. If question can be answered: direct answer

        Example:
        Input question: "华东区域的系列A总库存是多少?"
        - If no search has been done: <DB>华东区域的系列A总库存是多少?</DB>
        - If sufficient information exists: "374"
        """
        #print(global_config.get('query_type'))
        if global_config.get('query_type','NL')=='SQL':
            system_prompt=SYSTEM_PROMPT_SQL
        else:
            system_prompt=SYSTEM_PROMPT_NL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Question: " + query + "\n\n" + "Trace: " + str(trace)},
        ]

        response = oai.beta.chat.completions.parse(
            messages=messages,
            model="qwen-max-0125",
        )
        return response.choices[0].message.content ,response.usage.completion_tokens, response.usage.prompt_tokens
        # return response.choices[0].message.content

    def forward(self, **kwargs) -> dspy.Prediction:
        question = kwargs.get('question')

        if not question:
            raise ValueError("No question provided in input")

        trace = []
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'question': question,
            'steps': []
        }
        all_completions_tokens = 0
        all_prompt_tokens = 0

        start_time = time.time()

        for i in range(self.max_steps):
            # prediction = self.module(
            #     question=question,
            #     past_steps=self.format_trace(trace)
            # )
            # response = prediction.response
            response ,completion_tokens ,prompt_tokens = self.call_lm(question, trace)
            all_completions_tokens += completion_tokens
            all_prompt_tokens += prompt_tokens
            search_query = self.response_manager.extract_search_query(response)

            step_info = {
                'step': i + 1,
                'response': response
            }

            if search_query:
                search_result = self.response_manager.perform_db(search_query)
                # search_result = self.response_manager.perform_web_search_from_api(search_query)
                trace.append((search_query, search_result))
                step_info.update({
                    'search_query': search_query,
                    'search_result': search_result
                })
                log_entry['steps'].append(step_info)
            else:
                is_correct, ground_truth = self.evaluate_prediction(response, question)
                end_time = time.time()
                step_info.update({
                    'is_correct': is_correct,
                    'ground_truth': ground_truth,
                    'all_completion_tokens': all_completions_tokens,
                    'all_prompt_tokens': all_prompt_tokens,
                    'time_cost': end_time - start_time
                })
                log_entry['steps'].append(step_info)

                self.logger.info(json.dumps(log_entry, ensure_ascii=False))

                return dspy.Prediction(
                    answer=response,
                    trace=trace,
                    eval_report={
                        'success': is_correct,
                        'all_completion_tokens': all_completions_tokens,
                        'all_prompt_tokens': all_prompt_tokens,
                        'prediction': response,
                        'ground_truth': ground_truth
                    }
                )


        # 如果达到最大步数仍未得到答案
        end_time = time.time()
        log_entry['steps'].append({
            'step': self.max_steps,
            'response': "Reached maximum number of steps without finding an answer.",
            'is_correct': False,
            'ground_truth': "None",
            'all_completion_tokens': all_completions_tokens,
            'all_prompt_tokens': all_prompt_tokens,
            'time_cost': end_time - start_time
        })
        # 记录日志
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

        return dspy.Prediction(
            answer="Sorry, could not find an answer within the step limit",
            trace=trace,
            eval_report={
                'success': False,
                'all_completion_tokens': all_completions_tokens,
                'all_prompt_tokens': all_prompt_tokens,
                'prediction': "No answer found",
                'ground_truth': "None"
            }
        )
