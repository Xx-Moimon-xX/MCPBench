import json
import os

import dspy

from ..benchmark import Benchmark


class DBBench(Benchmark):
    def init_dataset(self):
        self.dataset = []  # 没有训练数据，设置为空列表

        test_path = os.path.join(os.path.dirname(__file__), 'data', 'car_bi.jsonl')
        self.test_set = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data = json.loads(line)
                # 将 Prompt 作为输入，Answer 作为输出
                self.test_set.append(
                    dspy.Example(
                        question=test_data["Prompt"],
                        answer=test_data["Answer"]
                    ).with_inputs("question")
                )
