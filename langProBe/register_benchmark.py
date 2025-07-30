########################## Benchmarks ##########################
import importlib


# To use registered benchmarks, do
# `benchmark.benchmark, benchmark.programs, benchmark.metric`
registered_benchmarks = []


def check_benchmark(benchmark):
    try:
        assert hasattr(benchmark, "benchmark")
    except AssertionError:
        return False
    return True


def register_benchmark(benchmark: str):
    '''
    This function is used to register a benchmark.
    Imports the benchmark and if it has a benchmark attribute, it adds it to the registered benchmarks.
    '''

    try:
        # Try to import the module directly
        benchmark_metas = importlib.import_module(benchmark, package="langProBe")
    except ModuleNotFoundError:
        # If direct import fails, try importing with full path
        benchmark_metas = importlib.import_module(f"langProBe.{benchmark}", package=None)
    
    # Adding it to the registered benchmarks (variable above)
    if check_benchmark(benchmark_metas):
        registered_benchmarks.extend(benchmark_metas.benchmark)
    else:
        raise AssertionError(f"{benchmark} does not have the required attributes")
    return benchmark_metas.benchmark


def register_all_benchmarks(benchmarks):
    '''
    Just calling register_benchmark for each benchmark in the list (all strings)
    '''
    for benchmark in benchmarks:
        register_benchmark(benchmark)
    return registered_benchmarks
