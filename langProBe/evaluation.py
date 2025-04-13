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
from langProBe.config_utils import read_json
from langProBe.dspy_program import (
    GeneratorCriticFuser,
    GeneratorCriticRanker,
    LangProBeDSPyMetaProgram,
)
from langProBe.optimizers import create_optimizer, DEFAULT_OPTIMIZERS
from langProBe.register_benchmark import register_all_benchmarks, registered_benchmarks


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
    file_path = pathlib.Path(file_path)

    with open(f"{file_path}/evaluation_records.csv", "a") as f:
        for evaluation_result in evaluation_results:
            f.write(
                f"{evaluation_result.benchmark},{evaluation_result.program},{evaluation_result.optimizer}\n"
            )


def read_evaluation_records(file_path):
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


program_class_mapping = {
    "baseline": [dspy.Predict],
    "single": [dspy.Predict, dspy.ChainOfThought],
    "archon": [GeneratorCriticFuser, GeneratorCriticRanker],
}


def evaluate(
    benchmark_meta: BenchmarkMeta,
    lm,
    rm,
    file_path,
    num_threads=8,
    suppress_dspy_output=True,
    dataset_mode=None,
    use_devset=False,
    missing_mode=False,
    program_class="all",
    api_key=None,
    api_base=None,
):
    """
    benchmark_meta: BenchmarkMeta object to evaluate
    lm: Language model to use, should be an instance of dspy.LM
    rm: Retrieval model to use
    optimizers: List[type(Teleprompter) | (type(Teleprompter), kwargs_for_compile)]
    missing_mode: only evaluate experiments without a result file
    program_class: the program class to evaluate
    """
    dataset_mode = dataset_mode or benchmark_meta.dataset_mode
    benchmark = benchmark_meta.benchmark(dataset_mode=dataset_mode)
    # Canonicalize optimizers to (optimizer, compile_kwargs) tuples
    optimizers = benchmark_meta.optimizers
    benchmark_name = benchmark_meta.name or benchmark.__class__.__name__

    num_threads = benchmark_meta.num_threads or num_threads
    print(f"Evaluating {benchmark_name}")
    print(f"Train set size: {len(benchmark.train_set)}")
    print(f"Validation set size: {len(benchmark.val_set)}")
    print(f"Test set size: {len(benchmark.test_set)}")

    optimizer_names = [optimizer.name for optimizer in optimizers]

    Path(file_path).mkdir(parents=True, exist_ok=True)

    evaluation_records = read_evaluation_records(file_path)

    # create a stats file for each experiment
    stats_file = os.path.join(file_path, f"{benchmark_name}.stat")
    with open(stats_file, "w") as f:
        f.write(
            f"benchmark: {benchmark_name}\n"
            f"lm: {lm}\n"
            f"rm: {rm}\n"
            f"train_set_size: {len(benchmark.train_set)}\n"
            f"val_set_size: {len(benchmark.val_set)}\n"
            f"test_set_size: {len(benchmark.test_set)}\n"
            f"optimizers: {optimizer_names}\n"
            f"optimizer_configs: {optimizers}\n"
        )

    for program in benchmark_meta.program:
        if program_class != "all":
            available_program_classes = program_class_mapping[program_class]
            # if not isinstance(program, tuple(available_program_classes)):
            #     continue

        program_name = getattr(program, "_name", program.__class__.__name__)

        evaluate_baseline_flag = True
        optimizers = copy.deepcopy(benchmark_meta.optimizers)
        if missing_mode:
            # Only run missing experiments
            for optimizer in benchmark_meta.optimizers:
                if (benchmark_name, program_name, optimizer.name) in evaluation_records:
                    optimizers.remove(optimizer)
            if (benchmark_name, program_name, "None") in evaluation_records:
                evaluate_baseline_flag = False

        print(f"Program: {program_name}, running baseline: {evaluate_baseline_flag}")
        if isinstance(program, LangProBeDSPyMetaProgram):
            print(f"Optimizers: {'; '.join(map(lambda x: x.name, optimizers))}")
        else:
            print("Evaluating baseline only for non-dspy programs.")

        with suppress_output(suppress=suppress_dspy_output):
            evaluate_bench = EvaluateBench(
                benchmark=benchmark,
                program=program,
                metric=benchmark_meta.metric,
                lm=lm,
                optimizers=[
                    create_optimizer(
                        optimizer,
                        benchmark_meta.metric,
                        num_threads=num_threads,
                    )
                    for optimizer in optimizers
                ],
                evaluate_baseline_flag=evaluate_baseline_flag,
                benchmark_name=benchmark_meta.name,
                num_threads=num_threads,
                use_devset=use_devset,
                api_key=api_key,
                api_base=api_base,
            )
            evaluate_bench.evaluate(dspy_config={"rm": rm})
        print(f"Results: {evaluate_bench.results}")

        if missing_mode:
            add_to_evaluation_records(file_path, evaluate_bench.results)
        # import pdb; pdb.set_trace()
        # logging all results
        for evaluation_result in evaluate_bench.results:
            file_name = f"{evaluation_result.benchmark}_{evaluation_result.program}_{evaluation_result.optimizer}"
            if evaluation_result.optimizer:
                optimizer_header = "optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens"
                optimizer_values = (
                    f"{evaluation_result.optimizer},{evaluation_result.optimizer_cost},"
                    f"{evaluation_result.optimizer_input_tokens},{evaluation_result.optimizer_output_tokens},"
                )
            else:
                optimizer_header = ""
                optimizer_values = ""
            # import pdb; pdb.set_trace()
            with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
                f.write(f"score,cost,input_tokens,output_tokens,{optimizer_header}\n")
                f.write(
                    f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
                    f"{evaluation_result.output_tokens},{optimizer_values}\n"
                )
            if evaluation_result.optimizer:
                evaluation_result.optimized_program.save(
                    os.path.join(file_path, f"{file_name}.json")
                )
            if evaluation_result.optimizer_program_scores:
                with open(
                    os.path.join(file_path, f"{file_name}_optimizer_score.txt"), "w"
                ) as f:
                    f.write(",".join(evaluation_result.optimizer_program_scores))


def evaluate_all(
    benchmarks,
    lm,
    rm,
    file_path,
    num_threads=8,
    suppress_dspy_output=False,
    dataset_mode=None,
    use_devset=False,
    missing_mode=False,
    program_class="all",
    api_key=None,
    api_base=None,
):
    # 只有当benchmarks是字符串列表时才进行注册
    if benchmarks and isinstance(benchmarks[0], str):
        benchmarks = register_all_benchmarks(benchmarks)
    if missing_mode:
        generate_evaluation_records(file_path)
    for benchmark_meta in benchmarks:
        evaluate(
            benchmark_meta,
            lm,
            rm,
            file_path,
            num_threads,
            suppress_dspy_output,
            dataset_mode,
            use_devset,
            missing_mode,
            program_class,
            api_key=api_key,
            api_base=api_base,
        )

    df = read_evaluation_results(file_path)
    df.to_csv(f"{file_path}/evaluation_results.csv", index=False)
    df["model"] = lm

    # generate evaluation records
    generate_evaluation_records(file_path)

global_config=None
def main():
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="LangProbe benchmark evaluation")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark to evaluate")
    parser.add_argument("--lm", type=str, required=True, help="Language model to use")
    parser.add_argument("--lm_api_key", type=str, help="API key for language model")
    parser.add_argument(
        "--lm_api_base", type=str, help="API base for language model"
    )
    parser.add_argument("--rm", type=str, help="Retrieval model to use")
    parser.add_argument(
        "--rm_api_key", type=str, help="API key for retrieval model"
    )
    parser.add_argument(
        "--dataset_mode", type=str, help="Dataset mode (train, val, test)"
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
        "--missing_mode",
        action="store_true",
        help="Only run missing experiments (skip experiments that already have results)",
    )
    parser.add_argument(
        "--use_devset",
        action="store_true",
        help="Use the dev set to optimize the program; otherwise, will use the train set",
    )
    parser.add_argument(
        "--program_class",
        type=str,
        default="all",
        choices=["all", "baseline", "single", "archon"],
        help="Which program classes to evaluate",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=DEFAULT_OPTIMIZERS,
        help="List of optimizers to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='ddgo.json',
        help="Configuration file for the benchmark",
    )
    args = parser.parse_args()

    global global_config
    global_config= read_json(args.config)
    if args.benchmark=='WebSearch':

        from langProBe.WebSearch import benchmark
    else:
        from langProBe.DB import benchmark
    # 处理benchmark参数
    benchmark_path = args.benchmark
    if not benchmark_path.startswith("langProBe."):
        # 如果不是以langProBe开头，添加前缀
        benchmark_path = f"langProBe.{benchmark_path}"
    
    # 注册所有基准测试
    register_all_benchmarks([benchmark_path])
    benchmark[0].program[0].add_logger(global_config['name'])
    benchmarks = [
        benchmark
        for benchmark in registered_benchmarks
        if (benchmark.name == args.benchmark or 
            benchmark.name == benchmark_path or
            benchmark.benchmark.__name__ == args.benchmark or 
            benchmark.benchmark.__name__ == benchmark_path.split('.')[-1])
    ]
    if not benchmarks:
        print(f"No benchmark registered with name {args.benchmark}")
        sys.exit(1)

    # set up dspy
    dspy.settings.configure(lm=args.lm, api_key=args.lm_api_key, api_base=args.lm_api_base)
    lm = dspy.settings.lm

    # set up rm
    rm = None

    # 运行评估
    start_time = time.time()
    evaluate_all(
        benchmarks,
        lm,
        rm,
        args.file_path,
        num_threads=args.num_threads,
        suppress_dspy_output=args.suppress_dspy_output,
        dataset_mode=args.dataset_mode,
        use_devset=args.use_devset,
        missing_mode=args.missing_mode,
        program_class=args.program_class,
        api_key=args.lm_api_key,
        api_base=args.lm_api_base,
    )
    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"评测完成，Avg. elapsed time: {total_elapsed/50}秒")

if __name__ == "__main__":
    import multiprocessing
    # multiprocessing.freeze_support()
    main()
