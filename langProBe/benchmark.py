"""
This module defines the core benchmarking framework for LangProBe. It provides abstract and concrete classes for handling datasets, running benchmarks, evaluating programs, and storing results/metadata. It also includes utility functions for language model setup and statistics calculation.
"""
import random, os, csv, json
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
from langProBe.program_utils import ProcessManager, build_system_content
import json
# from langProBe.utils import flatten_dict
from langProBe.split_evaluate import SplitEvaluate
import pdb


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
    def __init__(self, dataset_mode="lite", dataset_path=None, missing_data=[], source: str = "original"):
        '''
        Initializes MCPBench with a dataset mode, path, and optional missing data.
        '''
        self.dataset_path = dataset_path
        self.missing_data = missing_data
        # source: 'original' for the native JSONL, 'predictions' for predictions.jsonl from generate_only
        self.source = source
        super().__init__(dataset_mode=dataset_mode)

    def init_dataset(self):
        '''
        Loads the dataset and test set from the given path or missing_data, and creates dspy.Example objects for each entry.
        '''
        self.dataset = []
        self.test_set = []
        if self.source == "predictions":
            # predictions.jsonl format written by generate_only
            test_raw_data = read_jsonl(self.dataset_path)
            for row in test_raw_data:
                self.test_set.append(
                    dspy.Example(
                        id=row.get("id", ""),
                        question=row.get("question", ""),
                        ground_truth=row.get("ground_truth", ""),
                        answer=row.get("answer", ""),
                        tools_required=row.get("tools_required", []),
                        tools_called=row.get("tools_called", []),
                    ).with_inputs("id", "question", "ground_truth", "answer", "tools_required", "tools_called", "config")
                )
        else:
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
                        tools_required=test_data["tools_required"],
                    ).with_inputs("id", "question", "answer", "tools_required", "config")
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


"""
This module now ensures that for MCP tool calls, only one SyncedMcpClient process per server is created and reused for the entire evaluation run.
A persistent client_cache is created in EvaluateBench and passed to all tool calls (via mcp_calling), and all clients are cleaned up at the end.
"""
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
        config,
        benchmark_name: str = None,
        num_threads: int = 1,
        api_key: str = None,
        api_base: str = None,
        file_path=None,
        eval_lm=None,
        use_split: bool = False,
    ):
        '''
        Initializes the evaluation with the given benchmark, program, metric, language model, and other configs.
        '''
        self.benchmark = benchmark
        # Pass config to the program if it accepts it
        if hasattr(program, 'config'):
            program.config = config
        self.program = program
        # if hasattr(self.program, "set_dataset"):
        #     self.program.set_dataset(dataset)
        # if hasattr(self.program, "set_log_path"):
        self.program.set_log_path(file_path)
        self.program.setup_lm(lm, api_key=api_key, api_base=api_base)
        self.metric = metric
        self.num_threads = num_threads
        devset = benchmark.get_test_set()
        self.program.lm.eval_model = eval_lm
        # Split evaluator is available; run mode is controlled by evaluation.py
        self.file_path = file_path

        # --- Persistent client cache for MCP tool calls ---
        # This ensures only one SyncedMcpClient process per server for the entire evaluation run.
        from langProBe.program_utils import cleanup_all_clients
        self.client_cache = {}
        self._cleanup_all_clients = cleanup_all_clients
        # Always set the cache on the program if supported
        self.program.set_client_cache(self.client_cache)
        # --------------------------------------------------

        # Prepare both unified and split evaluators; choose at runtime via `use_split`.
        self.evaluate_prog = Evaluate(
            devset=devset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            max_errors=5000,
            return_outputs=True,
            provide_traceback=True,
        )
        self.split_eval = SplitEvaluate(
            devset=devset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=5000,
            provide_traceback=True,
        )
        # Keep a handle to devset for split workflows (generate/score only)
        self.devset = devset
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
        # Patch: inject client_cache into the program if it supports it
        if hasattr(self.program, 'set_client_cache'):
            self.program.set_client_cache(self.client_cache)
        elif hasattr(self.program, 'client_cache'):
            self.program.client_cache = self.client_cache
        # else: for legacy programs, they must pass client_cache manually to mcp_calling

        with dspy.context(**(dspy_config or {})):
            score, info = self.evaluate_prog(self.program)

        result = self.get_empty_results()
        datasets, outputs, _ = zip(*info)
        managers = [getattr(one, 'process_report', None) for one in outputs]
        managers = [m for m in managers if m is not None]

        result.score = score   
        result.outputs_raw_data = outputs
        result.cost, result.input_tokens, result.output_tokens = calculate_stats(managers)

        # --- Clean up all MCP clients at the end ---
        self._cleanup_all_clients(self.client_cache)
        # ------------------------------------------

        return result

    def evaluate_baseline_split(self, dspy_config=None) -> EvaluationResult:
        '''
        Evaluates in two stages: generate first, then score.
        '''
        # Patch: inject client_cache into the program if it supports it
        if hasattr(self.program, 'set_client_cache'):
            self.program.set_client_cache(self.client_cache)
        elif hasattr(self.program, 'client_cache'):
            self.program.client_cache = self.client_cache

        with dspy.context(**(dspy_config or {})):
            # Ensure generate-only mode to bypass in-forward evaluation
            if hasattr(self.program, 'set_run_mode'):
                self.program.set_run_mode('generate_only')
            predictions = self.split_eval.generate(program=self.program, display_progress=True)
            if hasattr(self.program, 'set_run_mode'):
                self.program.set_run_mode('combined')
            score, detailed = self.split_eval.score(predictions)

        result = self.get_empty_results()
        # detailed is [(example, prediction, score_float)], we only need predictions for downstream CSV
        outputs = [pred for (_, pred, _) in detailed]
        managers = [getattr(pred, 'process_report', None) for pred in outputs]
        managers = [m for m in managers if m is not None]

        result.score = score
        result.outputs_raw_data = outputs
        result.cost, result.input_tokens, result.output_tokens = calculate_stats(managers)

        # --- Clean up all MCP clients at the end ---
        self._cleanup_all_clients(self.client_cache)
        # ------------------------------------------

        return result

    # --- Standalone generate/score helpers ---
    def _responses_dir(self) -> str:
        base = self.file_path or "."
        path = os.path.join(base, "response_data")
        os.makedirs(path, exist_ok=True)
        return path

    def _responses_csv_path(self) -> str:
        return os.path.join(self._responses_dir(), "predictions.csv")

    def _responses_jsonl_path(self) -> str:
        return os.path.join(self._responses_dir(), "predictions.jsonl")

    def _save_responses_to_csv(self, predictions: list) -> str:
        """Save (example, prediction) pairs into response_data/predictions.csv.
        Stores evaluation_data as JSON string for fidelity.
        """
        csv_path = self._responses_csv_path()
        headers = [
            "id",
            "question",
            "ground_truth",
            "answer",
            "tools_required", 
            "tools_called"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for example, pred in zip(self.devset, predictions):
                # Extract fields robustly
                q = getattr(example, "question", "")
                gt = getattr(example, "answer", getattr(example, "ground_truth", ""))
                ans = getattr(pred, "answer", "")
                tools_required = getattr(example, "tools_required", "")
                tools_called = getattr(pred, "tools_called", "")
                # tcs = getattr(pred, "tool_calling_success", "")
                # suc = getattr(pred, "success", "")
                # ed = getattr(pred, "evaluation_data", None)

                breakpoint()

                writer.writerow([
                    getattr(example, "id", ""),
                    q,
                    gt,
                    ans,
                    tools_required,
                    tools_called,
                ])
        return csv_path

    def _serialize_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _serialize_tools_called(self, tools_called):
        if not tools_called:
            return []
        if isinstance(tools_called, list):
            serialized = []
            for call in tools_called:
                name = getattr(call, 'mcp_tool_name', None)
                serialized.append(name if name is not None else str(call))
            return serialized
        return [str(tools_called)]

    def _save_responses_to_jsonl(self, predictions: list) -> str:
        """Save (example, prediction) pairs into response_data/predictions.jsonl."""
        jsonl_path = self._responses_jsonl_path()
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example, pred in zip(self.devset, predictions):
                record = {
                    "id": getattr(example, "id", ""),
                    "question": getattr(example, "question", ""),
                    "ground_truth": getattr(example, "answer", getattr(example, "ground_truth", "")),
                    "answer": getattr(pred, "answer", ""),
                    "tools_required": self._serialize_list(getattr(example, "tools_required", [])),
                    "tools_called": self._serialize_tools_called(getattr(pred, "tools_called", [])),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonl_path

    def generate_and_save_responses(self, dspy_config=None) -> str:
        """Generate predictions only and save to CSV. Returns the CSV path."""
        if hasattr(self.program, 'set_client_cache'):
            self.program.set_client_cache(self.client_cache)
        elif hasattr(self.program, 'client_cache'):
            self.program.client_cache = self.client_cache
        
        print(f"In generate and save responses\n")
        # pdb.set_trace(header="in generate and save responses")
        breakpoint()

        with dspy.context(**(dspy_config or {})):
            # Force generate-only mode so program.forward skips internal evaluation
            if hasattr(self.program, 'set_run_mode'):
                self.program.set_run_mode('generate_only')
            # pdb.set_trace(header="before generating with split_eval")
            breakpoint()

            predictions = self.split_eval.generate(program=self.program, display_progress=True)

            # pdb.set_trace(header="after generating with split_eval")
            breakpoint()

            if hasattr(self.program, 'set_run_mode'):
                self.program.set_run_mode('combined')

        csv_path = self._save_responses_to_csv(predictions)
        self._save_responses_to_jsonl(predictions)
        # Clean up clients
        self._cleanup_all_clients(self.client_cache)
        return csv_path

    def _load_responses_from_csv(self) -> list:
        """Load predictions from CSV and align to devset order using example id."""
        csv_path = self._responses_csv_path()
        by_id = {}
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Responses file not found: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pred = dspy.Prediction()
                pred.question = row.get("question", "")
                pred.ground_truth = row.get("ground_truth", "")
                pred.answer = row.get("answer", "")


                # pred.tool_calling_success = row.get("tool_calling_success", "")
                # pred.success = row.get("success", "")
                # ed_json = row.get("evaluation_data_json", "")
                try:
                    pred.evaluation_data = json.loads(ed_json) if ed_json else None
                except json.JSONDecodeError:
                    pred.evaluation_data = None
                by_id[row.get("id", "")] = pred

        aligned = []
        for example in self.devset:
            key = getattr(example, "id", "")
            aligned.append(by_id.get(key, dspy.Prediction()))
        return aligned

    def score_from_saved_responses(self, dspy_config=None) -> EvaluationResult:
        """Score previously generated predictions loaded from CSV."""
        # Prefer program-native score_only path so per-example evaluators are used
        if hasattr(self.program, 'set_run_mode'):
            self.program.set_run_mode('score_only')

        from dspy.utils.parallelizer import ParallelExecutor

        preds = []
        scores = []
        with dspy.context(**(dspy_config or {})):
            # Show live average via compare_results=True
            executor = ParallelExecutor(
                num_threads=self.num_threads,
                disable_progress_bar=False,
                max_errors=5000,
                provide_traceback=True,
                compare_results=True,
            )

            def process_item(example):
                prediction = self.program(**example.inputs())
                score_val = self.metric(example, prediction)
                return prediction, float(score_val)

            results = executor.execute(process_item, self.devset)

        if hasattr(self.program, 'set_run_mode'):
            self.program.set_run_mode('combined')

        # Aggregate
        total = 0.0
        count = 0
        for item in results:
            if item is None:
                preds.append(dspy.Prediction())
                continue
            prediction, score_val = item
            preds.append(prediction)
            total += score_val
            count += 1

        avg_percent = round(100.0 * total / count, 2) if count else 0.0
        result = self.get_empty_results()
        result.score = avg_percent
        result.outputs_raw_data = preds
        result.cost, result.input_tokens, result.output_tokens = (0, 0, 0)
        return result

    def evaluate(self, dspy_config=None) -> EvaluationResult:
        '''
        Evaluates the program on the benchmark (optionally with config) and returns an EvaluationResult.
        '''
        if dspy_config is None:
            dspy_config = {}

        if self.use_split:
            result = self.evaluate_baseline_split(dspy_config)
        else:
            result = self.evaluate_baseline(dspy_config)
        self.results = result
        return result
