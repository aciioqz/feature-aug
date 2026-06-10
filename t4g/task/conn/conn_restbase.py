import pandas as pd

from t4g.data import BenchTask
from t4g.data.task import TaskType
from t4g.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc, macro_f1, micro_f1, mse, r2


class ConnRestbaseClasTask(BenchTask):

    name = "conn-restbase-classification"
    task_type = TaskType.REGRESSION
    entity_col = "id_restaurant"
    entity_table = "base"
    target_col = "review"
    metrics = [mae, mse, rmse, r2]