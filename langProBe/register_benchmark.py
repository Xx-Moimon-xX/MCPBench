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
    try:
        # 尝试直接导入模块
        # Try to import the module directly
        benchmark_metas = importlib.import_module(benchmark, package="langProBe")
    except ModuleNotFoundError:
        # 如果直接导入失败，尝试使用完整路径导入
        # If direct import fails, try importing with full path
        benchmark_metas = importlib.import_module(f"langProBe.{benchmark}", package=None)
    
    if check_benchmark(benchmark_metas):
        registered_benchmarks.extend(benchmark_metas.benchmark)
    else:
        raise AssertionError(f"{benchmark} does not have the required attributes")
    return benchmark_metas.benchmark


def register_all_benchmarks(benchmarks):
    for benchmark in benchmarks:
        register_benchmark(benchmark)
    return registered_benchmarks
