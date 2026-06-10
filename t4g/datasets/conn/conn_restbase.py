import os

import pandas as pd
from t4g.data import BenchTask
from t4g.data import Database, BenchDataset, Table
from t4g.task.conn.conn_restbase import ConnRestbaseClasTask
from t4g.util.utils import to_unix_time, unzip_processor
import numpy as np
from typing import Dict
from t4g.datasets.conn.utlis import create_database


def drop_column(table_df, users_drop_columns):
    for column in users_drop_columns:
        try:
            table_df.drop(columns=[column], inplace=True)
        except KeyError:
            pass


class SelectedConnRestbaseDataset(BenchDataset):
    name = "selected-conn-restbase"
    task_cls_list = [ConnRestbaseClasTask]

    def __init__(
            self,
            *,
            process: bool = False,
            path: str
    ):
        self.name = f"{self.name}"
        self.path = path
        super().__init__(process=process, path=path)

    def make_db(self) -> Database:
        path = self.path
        path = path.replace("conn-", "")  
        print(f"path:{path}")
        task: BenchTask = self.task_cls_list[0]
        return create_database(path, task)


