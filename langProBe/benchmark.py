"""
This module defines the core benchmarking framework for LangProBe. It provides abstract and concrete classes for handling datasets, running benchmarks, evaluating programs, and storing results/metadata. It also includes utility functions for language model setup and statistics calculation.
"""
import random, os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Type

import dspy
from dspy.evaluate import Evaluate
# from dspy.teleprompt import Teleprompter

import langProBe.optimizers as langprobe_optimizers
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from langProBe.config_utils import read_json, read_jsonl
from langProBe.program_utils import ProcessManager


"""
Main class for benchmark evaluation and starting it with the DSPy Evaluate class.

This file has: 
- Benchmark, EvaluateBench, MCPBench classes.
- BenchmarkMeta, EvaluationResult are the data classes.
- Individual functions setup_lm and calculate_stats
"""


dataset_size = {"full": None, "lite": 500, "tiny": 200, "test": 2}


class Benchmark(ABC):
    '''
    Abstract base class for benchmarks. Handles dataset loading, splitting, and provides access to train/dev/test sets.
    '''
    def __init__(self, dataset_mode="lite"):
        '''
        Initializes the benchmark, loads and splits the dataset according to the mode, and sets up train/dev/test sets.
        '''
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.test_set = None
        self.train_set = None
        self.dev_set = None
        self.val_set = None

        self.init_dataset()
        assert self.dataset is not None, "Dataset not initialized"
        assert self.test_set is not None, "Test set not initialized"
        self.max_testset_size = dataset_size[dataset_mode]

        self.test_set = self.trim_dataset(self.test_set, self.max_testset_size)

        # TODO: FIXME: "test" option is for debugging purposes only, should be removed for final release
        if dataset_mode == "test":
            self.dataset = self.trim_dataset(self.dataset, 60)
            self.create_splits()
            self.test_set = self.trim_dataset(self.test_set, 50)

        if not self.train_set or not self.dev_set or not self.val_set:
            self.create_splits()

        self.train_set = self.trim_dataset(self.train_set, 150)
        self.dev_set = self.trim_dataset(self.dev_set, 300)
        self.val_set = self.trim_dataset(self.val_set, 300)

        assert self.train_set is not None, "Train set not initialized"
        assert self.dev_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Abstract method to initialize the dataset. Must be implemented by subclasses.
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        return

    def trim_dataset(self, dataset, size: int) -> None:
        '''
        Randomly samples up to 'size' items from the dataset, or returns the full dataset if size is None or too large.
        '''
        if size is None or size >= len(dataset):
            return dataset
        rng = random.Random()
        rng.seed(1)
        return rng.sample(dataset, size)

    def create_splits(self) -> None:
        """
        Splits the dataset into dev, val, and train sets (not including test set).
        Creates the splits for the dataset (not including test).
        Upon completion, self.train_set, self.dev_set, and self.val_set should be set.
        """

        total_len = len(self.dataset)
        self.dev_set = self.dataset[: int(0.4 * total_len)]
        self.val_set = self.dataset[int(0.4 * total_len) : int(0.8 * total_len)]
        self.train_set = self.dataset[int(0.8 * total_len) :]

    def get_dataset(self):
        '''Returns the full dataset.'''
        return self.dataset

    def get_train_set(self):
        '''Returns the training set.'''
        return self.train_set

    def get_dev_set(self):
        '''Returns the development set.'''
        return self.dev_set

    def get_test_set(self):
        '''Returns the test set.'''
        return self.test_set


class MCPBench(Benchmark):
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
                ).with_inputs("id", "question", "answer", "config")
            )

# @dataclass is a Python decorator that automatically generates special methods like __init__, __repr__, and __eq__ for classes that are mainly used to store data.
@dataclass
class EvaluationResult:
    '''
    Stores the results of a single evaluation, including benchmark/program names, score, cost, token usage, and optional raw outputs.
    '''
    benchmark: str
    program: str

    score: float
    cost: float
    input_tokens: int
    output_tokens: int

    outputs_raw_data: List|None = None

    # optimizer: str = None
    # optimized_program: dspy.Module = None
    # optimizer_input_tokens: int = None
    # optimizer_output_tokens: int = None
    # optimizer_cost: float = None

    # optimizer_program_scores: list[float] = None


@dataclass
class BenchmarkMeta:
    '''
    Stores metadata for a benchmark, including the benchmark class, program(s), metric, dataset mode, optimizers, threading, and name.
    '''
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "lite"

    optimizers: List[langprobe_optimizers.OptimizerConfig] = field(
        default_factory=lambda: langprobe_optimizers.DEFAULT_OPTIMIZERS
    )

    # BenchmarkMeta.num_threads has higher priority than run time argument of num_threads
    # use this as an upper bound for the number of threads to use
    num_threads: int = None
    name: str = None

    def __repr__(self):
        return (f"<BenchmarkMeta(benchmark={repr(self.benchmark)}, "
                f"program={repr(self.program)}, "
                f"metric={repr(self.metric)}, "
                f"dataset_mode={repr(self.dataset_mode)}, "
                f"optimizers={repr(self.optimizers)}, "
                f"num_threads={repr(self.num_threads)}, "
                f"name={repr(self.name)})>")


def setup_lm(dspy_config=None):
    '''
    Sets up and returns a copy of the dspy language model (LM) from the given config, ensuring it has no history.
    '''
    lm: dspy.LM = dspy_config.get("lm", dspy.settings.lm)
    assert lm is not None, "dspy language model not set"

    lm = lm.copy()
    assert len(lm.history) == 0, "language model history not empty"
    return lm


# def calculate_stats(lm: dspy.LM) -> tuple[float, int, int]:
#     cost = 0
#     input_tokens = 0
#     output_tokens = 0
#     for i, trace in enumerate(lm.history):
#         cost += trace.get("cost", None) or 0
#         input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
#         output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

#     return cost, input_tokens, output_tokens

def calculate_stats(manager: List[ProcessManager]) -> tuple[float, float, float]:
    '''
    Calculates and returns (dummy cost, average input tokens, average output tokens) from a list of ProcessManager objects.
    '''
    input_tokens = sum(usage["prompt_tokens"] for trace in manager for usage in trace.lm_usages)
    output_tokens = sum(usage["completion_tokens"] for trace in manager for usage in trace.lm_usages)
    
    avg_input = input_tokens // len(manager)
    avg_output = output_tokens // len(manager)
    
    return 0, avg_input, avg_output



class EvaluateBench(ABC):
    '''
    Abstract base class for evaluating a program on a benchmark. Handles evaluation logic, result storage, and metric calculation.
    '''
    def __init__(
        self,
        benchmark: Benchmark,
        program: dspy.Module,
        metric: Callable,
        lm: str,
        benchmark_name: str = None,
        num_threads: int = 1,
        api_key: str = None,
        api_base: str = None,
        config=None,
    ):
        '''
        Initializes the evaluation with the given benchmark, program, metric, language model, and other configs.
        '''

        self.benchmark = benchmark
        # Pass config to the program if it accepts it
        if hasattr(program, 'config'):
            program.config = config
        self.program = program
        self.program.setup_lm(lm, api_key=api_key, api_base=api_base)
        self.metric = metric
        self.num_threads = num_threads
        devset = benchmark.get_test_set()


        # Everything is done inside here!!
        # Devset is just the json file with given input and output. Example below::

        # DEBUG: devset: [Example({'id': 676, 'question': 'What is the country of origin of the football coach with the first initial "P" for the Thailand national men\'s football team who coached 54 years after the country\'s name officially changed?', 'answer': 'German.'}) (input_keys={'question', 'config', 'answer', 'id'}), Example({'id': 537, 'question': 'I am thinking of a movie where Hans Zimmer won a Grammy Award for his work. He won the Grammy award the same year that he did his first musical score for film director Michael Bay. Can you please tell me the name of that movie?', 'answer': 'Crimson Tide'}) (input_keys={'question', 'config', 'answer', 'id'})]
        
        # Looks like the response generation and evaluation is all done together here..??
        # Evaluate calls our program(**example.input) function i.e. the forward pass to geth the prediction.
        # Response generation is done by your program/model (e.g., MCPPredict), not by the metric or the high-level evaluate function.
        # The Evaluate class orchestrates the process: for each example, it calls your program to generate a response, then calls the metric to score it.
        
        # Black box, does it all for you, this is the DSPy Evaluate class, I understand how it works now.
        self.evaluate_prog = Evaluate(
            devset=devset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            max_errors=5000,
            return_outputs=True,
            provide_traceback=True,
        )
        self.program_name = getattr(
            self.program, "_name", self.program.__class__.__name__
        )
        self.benchmark_name = benchmark_name or self.benchmark.__class__.__name__
        self.results: list[EvaluationResult] = []

    def __repr__(self):
        return (f"<EvaluateBench(benchmark={repr(self.benchmark)}, "
                f"program={repr(self.program)}, "
                f"metric={repr(self.metric)}, "
                f"lm={repr(self.program.lm) if hasattr(self.program, 'lm') else None}, "
                f"benchmark_name={repr(self.benchmark_name)}, "
                f"num_threads={self.num_threads}, "
                f"results={repr(self.results)})>")

    def get_empty_results(self):
        '''
        Returns an empty EvaluationResult object for this evaluation.
        '''
        return EvaluationResult(
            benchmark=self.benchmark_name,
            program=self.program_name,
            score=0,
            cost=0,
            input_tokens=0,
            output_tokens=0,
        )

    def evaluate_baseline(self, dspy_config=None) -> EvaluationResult:
        '''
        Evaluates the program on the benchmark using the baseline method and returns an EvaluationResult.
        '''

        ## Everything is not done here, it's further in evaluate_prog(self.program)
        with dspy.context(**dspy_config):
            # Program is passed to Evaluate class here.
            # info containe tuple: (example, prediction, score)
            score, info = self.evaluate_prog(self.program)
            # print(f"DEBUG: score: {score} info: {info}")

        result = self.get_empty_results()
        datasets, outputs, _ = zip(*info)
        # print(f"DEBUG: datasets: {datasets} outputs: {outputs}")
        managers = [getattr(one, 'process_report', None) for one in outputs]
        managers = [m for m in managers if m is not None]

        result.score = score   
        result.outputs_raw_data = outputs
        result.cost, result.input_tokens, result.output_tokens = calculate_stats(managers)

        # print(f"DEBUG: result: {result}")
        return result

    def evaluate(self, dspy_config=None) -> EvaluationResult:
        '''
        Evaluates the program on the benchmark (optionally with config) and returns an EvaluationResult.
        '''
        if dspy_config is None:
            dspy_config = {}

        result = self.evaluate_baseline(dspy_config)
        self.results = result
        return result
