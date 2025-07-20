import argparse
import copy
import os
import pathlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import dspy

from langProBe.analysis import read_evaluation_results
from langProBe.benchmark import BenchmarkMeta, EvaluateBench, EvaluationResult
from langProBe.config_utils import read_json, read_jsonl
from langProBe.dspy_program import (
    GeneratorCriticFuser,
    GeneratorCriticRanker,
    LangProBeDSPyMetaProgram,
)
from langProBe.optimizers import create_optimizer, DEFAULT_OPTIMIZERS
from langProBe.register_benchmark import register_all_benchmarks, registered_benchmarks
from langProBe.evaluation_utils import find_missing_entries, replace_logger_filehandler

# Global configuration variable that can be accessed by other modules
global_config = None

"""
This file is the starting point and has the evaluate() and evaluate_all() functions.
- Deals with config file here and the command line arguments. 
- It also with all the evaluation_records.csv, evaluation_results.csv and MCP_WEBSEARCH.stat stuff. 
"""

class CompareAnswerSignature(dspy.Signature):
    """
    Compare the answer to the ground truth answer.
    """

    answer = dspy.InputField(desc="The answer to a problem")
    ground_truth = dspy.InputField(desc="The ground truth answer to the same problem")
    is_correct = dspy.OutputField(
        desc="Whether the answer is correct, either True or False."
    )


class CompareAnswer(dspy.Module):
    def __init__(self):
        self.compare_answer = dspy.ChainOfThought(CompareAnswerSignature)

    def forward(self, ground_truth, answer):
        pred = self.compare_answer(answer=answer, ground_truth=ground_truth)
        return pred


def llm_as_judge_evaluate(gold, pred, extract_answer_fun=lambda x: x.answer):
    compare_answer = CompareAnswer()
    answer_raw = compare_answer(
        ground_truth=extract_answer_fun(gold), answer=extract_answer_fun(pred)
    ).is_correct
    if answer_raw.lower().startswith("true"):
        return True
    else:
        return False


@contextmanager
def suppress_output(suppress=True):
    if suppress:
        # Save the original streams
        original_stderr = sys.stderr
        original_stdout = sys.stdout

        # Redirect stderr and stdout to devnull
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    try:
        yield
    finally:
        if suppress:
            # Restore the original streams
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = original_stderr
            sys.stdout = original_stdout


def generate_evaluation_records(file_path):
    '''
    Generate the evaluation records from the result files. 
    This doesn't seem to really work, idk what it's doing.
    '''
    file_path = pathlib.Path(file_path)

    # if the records file already exists, do not overwrite it
    if (file_path / "evaluation_records.csv").exists():
        return

    # List all .txt files in the directory
    all_result_files = list(file_path.rglob("*.txt"))

    records = []

    # Process each file
    for file in all_result_files:
        # Split the filename to get benchmark, program, and optimizer
        file_name_parts = file.stem.split("_")
        if len(file_name_parts) >= 3:
            benchmark = file_name_parts[0]
            program = file_name_parts[1]
            optimizer = file_name_parts[2]
            records.append((benchmark, program, optimizer))
        else:
            raise ValueError(f"Invalid file name: {file.name}")

    with open(f"{file_path}/evaluation_records.csv", "w") as f:
        f.write("benchmark,program,optimizer\n")
        for record in records:
            f.write(",".join(record) + "\n")


def add_to_evaluation_records(file_path, evaluation_results: list[EvaluationResult]):
    '''
    Add the evaluation results to the evaluation records file.
    Also not really working properly.
    '''
    file_path = pathlib.Path(file_path)

    with open(f"{file_path}/evaluation_records.csv", "a") as f:
        for evaluation_result in evaluation_results:
            f.write(
                f"{evaluation_result.benchmark},{evaluation_result.program},{evaluation_result.optimizer}\n"
            )


def read_evaluation_records(file_path):
    '''
    Reads the evaluation records from the file path.
    If the file does not exist, it creates an empty file with a header, otherwise it reads the file and returns each record as a tuple.
    '''
    file_path = pathlib.Path(file_path)
    records = []

    # create the records file if it does not exist
    if not (file_path / "evaluation_records.csv").exists():
        # create empty records file without header
        with open(f"{file_path}/evaluation_records.csv", "w") as f:
            f.write("")
    with open(f"{file_path}/evaluation_records.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            records.append(tuple(line.strip().split(",")))

    return records


def evaluate(
    benchmark_meta: BenchmarkMeta,
    lm,
    file_path,
    num_threads=8,
    suppress_dspy_output=True,
    dataset_mode=None,
    dataset_path=None,
    missing_mode_file="",
    api_key=None,
    api_base=None,
    config=None,
):
    """
    benchmark_meta: BenchmarkMeta object to evaluate
    lm: Language model to use, should be an instance of dspy.LM
    missing_mode: only evaluate experiments without a result file
    """

    # If the dataset mode is not provided, use the dataset mode from the benchmark meta
    dataset_mode = dataset_mode or benchmark_meta.dataset_mode

    # If the missing mode file is provided, we use it to find the missing data
    if missing_mode_file:
        origin_data = read_jsonl(dataset_path)
        runed_data = read_jsonl(missing_mode_file)
        missing_data = find_missing_entries(origin_data, runed_data)
        benchmark = benchmark_meta.benchmark(dataset_mode=dataset_mode, missing_data=missing_data)
        replace_logger_filehandler(os.path.splitext(missing_mode_file)[0])
    else:
        benchmark = benchmark_meta.benchmark(dataset_mode=dataset_mode, dataset_path=dataset_path)

    # Canonicalize optimizers to (optimizer, compile_kwargs) tuples
    benchmark_name = benchmark_meta.name or benchmark.__class__.__name__

    # If the number of threads is not provided, use the number of threads from the benchmark meta
    num_threads = benchmark_meta.num_threads or num_threads
    print(f"evaluating {benchmark_name}")
    print(f"num_threads: {num_threads}")
    print(f"test set size: {len(benchmark.test_set)}")

    # Create the file path for the evaluation results if it does not exist
    Path(file_path).mkdir(parents=True, exist_ok=True)

    # Read the evaluation records from the file path or if no file creates a new one
    evaluation_records = read_evaluation_records(file_path)

    # create a stats file for each experiment and writes into it all the metadata for the experiment
    stats_file = os.path.join(file_path, f"{benchmark_name}.stat")
    with open(stats_file, "w") as f:
        f.write(
            f"benchmark: {benchmark_name}\n"
            f"lm: {lm}\n"
            f"test_set_size: {len(benchmark.test_set)}\n"
        )

    

    # For each program in the benchmark, we evaluate it
    for program in benchmark_meta.program:
        program_name = getattr(program, "_name", program.__class__.__name__)
        ## suppress_output is a context manager that suppresses the output of the program.
        with suppress_output(suppress=suppress_dspy_output):

            # Initialising the evaluation benchmark.
            # THIS IS WHERE THE PROGRAM IS PASSED TO THE EVALUATE BENCH FUNCTION.
            evaluate_bench = EvaluateBench(
                benchmark=benchmark,
                program=program,
                metric=benchmark_meta.metric,
                lm=lm,
                benchmark_name=benchmark_meta.name,
                num_threads=num_threads,
                api_key=api_key if api_key else os.getenv("AWS_ACCESS_KEY_ID", ""),
                api_base=api_base if api_base else "",
                config=config,
            )
            evaluate_bench.evaluate()

        # if missing_mode:
        #     add_to_evaluation_records(file_path, evaluate_bench.results)
        evaluation_result = evaluate_bench.results

        file_name = f"{evaluation_result.benchmark}_{evaluation_result.program}"
        # TO DO: might want to change the output here for how we want it displayed.
        with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
            f.write(f"score,cost,input_tokens,output_tokens\n")
            f.write(
                f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
                f"{evaluation_result.output_tokens}\n"
            )


def evaluate_all(
    benchmarks,
    lm,
    file_path,
    num_threads=8,
    suppress_dspy_output=False,
    dataset_mode=None,
    dataset_path=None,
    missing_mode_file="",
    api_key=None,
    api_base=None,
    config=None,
):
    # Only register when benchmarks is a list of strings
    if benchmarks and isinstance(benchmarks[0], str):
        benchmarks = register_all_benchmarks(benchmarks)

    for benchmark_meta in benchmarks:
        evaluate(
            benchmark_meta,
            lm,
            file_path,
            num_threads,
            suppress_dspy_output,
            dataset_mode,
            dataset_path,
            missing_mode_file,
            api_key=api_key,
            api_base=api_base,
            config=config,
        )

    # After all the evaluations are done, we read the evaluation results and save them to a csv file
    df = read_evaluation_results(file_path)
    df.to_csv(f"{file_path}/evaluation_results.csv", index=False)
    df["model"] = lm

    # generate evaluation records 
    generate_evaluation_records(file_path)


def main():
    import multiprocessing
    multiprocessing.freeze_support()

    ## Exctracting all the arguments from the command line
    parser = argparse.ArgumentParser(description="LangProbe benchmark evaluation")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to evaluate")
    parser.add_argument("--lm", type=str, required=True, help="Language model to use")
    parser.add_argument("--lm_api_key", type=str, help="API key for language model")
    parser.add_argument(
        "--lm_api_base", type=str, help="API base for language model"
    )
    parser.add_argument(
        "--dataset_mode", type=str, help="Dataset mode (train, val, test)"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Dataset path"
    )
    parser.add_argument(
        "--num_threads", type=int, default=8, help="Number of threads to use"
    )
    parser.add_argument(
        "--file_path", type=str, default="evaluation", help="File path for evaluation results"
    )
    parser.add_argument(
        "--suppress_dspy_output",
        action="store_true",
        help="Suppress dspy output",
    )
    parser.add_argument(
        "--missing_mode_file",
        type=str,
        default="",
        help="Only run missing experiments (skip experiments that already have results), value = path to log/jsonl",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='ddgo.json',
        help="Configuration file for the benchmark",
    )

    args = parser.parse_args()

    config = read_json(args.config)
    
    # Set global config for use by other modules
    global global_config
    global_config = config

    # Process benchmark parameter
    benchmark_path = args.benchmark
    if not benchmark_path.startswith("langProBe."):
        benchmark_path = f"langProBe.{benchmark_path}"
    
    # Register all benchmarks
    # Basically just importing the websearch/DB/GAIA python modules
    register_all_benchmarks([benchmark_path])

    
    benchmarks = [benchmark for benchmark in registered_benchmarks]
    if not benchmarks:
        print(f"No benchmark registered with name {args.benchmark}\n")
        sys.exit(1)

    evaluate_all(
        benchmarks,
        args.lm,
        args.file_path,
        num_threads=args.num_threads,
        suppress_dspy_output=args.suppress_dspy_output,
        dataset_mode=args.dataset_mode,
        dataset_path=args.dataset_path,
        missing_mode_file=args.missing_mode_file,
        api_key=args.lm_api_key,
        api_base=args.lm_api_base,
        config=config,
    )

if __name__ == "__main__":
    main()
