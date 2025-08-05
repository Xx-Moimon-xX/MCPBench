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


def register_all_benchmarks(benchmarks, config=None):
    '''
    Registers all benchmarks in the list.
    For each benchmark module, tries to call a getter function with config if available,
    otherwise falls back to the 'benchmark' attribute.
    '''
    import importlib
    registered = []
    getter_names = [
        "get_eval_benchmark_1",
        "get_eval_benchmark_2",
        "get_eval_benchmark_3",
        "get_mcp_sample_benchmark",  # generic
        # Add more as needed
    ]
    for benchmark in benchmarks:
        module = importlib.import_module(benchmark)
        for getter in getter_names:
            if hasattr(module, getter):
                registered.extend(getattr(module, getter)(config))
                break
        else:
            # Fallback: use the 'benchmark' attribute if present (as in register_benchmark)
            if hasattr(module, "benchmark"):
                bench = getattr(module, "benchmark")
                # If it's a list, extend; if not, append
                if isinstance(bench, list):
                    registered.extend(bench)
                else:
                    registered.append(bench)
            else:
                raise AssertionError(f"{benchmark} does not have a recognized getter or 'benchmark' attribute")
    return registered
