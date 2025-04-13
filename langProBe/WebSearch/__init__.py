from langProBe.benchmark import BenchmarkMeta
from .WebSearch_utils.websearch_metric import websearch_metric
from .websearch_data import WebSearchBench
from .websearch_program import WebSearchPredict


# 延迟初始化，避免在导入时创建实例
def get_websearch_benchmark():
    websearch_student = WebSearchPredict()
    return [
        BenchmarkMeta(
            WebSearchBench,
            [websearch_student],
            websearch_metric,
            optimizers=[],
            name="WebSearch"  # 添加显式名称
        )
    ]

# 初始化变量以供外部访问
benchmark = get_websearch_benchmark()  # 直接调用get_websearch_benchmark初始化benchmark