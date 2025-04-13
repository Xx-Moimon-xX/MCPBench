from langProBe.benchmark import BenchmarkMeta
from .DB_utils.db_metric import db_metric
from .db_data import DBBench
from .db_program import DBPredict


# 延迟初始化，避免在导入时创建实例
def get_db_benchmark():
    db_student = DBPredict()
    return [
        BenchmarkMeta(
            DBBench,
            [db_student],
            db_metric,
            optimizers=[],
            name="DB"  # 添加显式名称
        )
    ]

# 初始化变量以供外部访问
benchmark = get_db_benchmark()  # 直接调用get_db_benchmark初始化benchmark