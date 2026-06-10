import os

import pandas as pd

from t4g.data import Database, BenchDataset, Table
from t4g.util.utils import to_unix_time, unzip_processor
import numpy as np
from typing import Dict
from t4g.data import BenchTask
from t4g.data.task import TaskType

def create_database(path:str, task:BenchTask):
    conn_path = os.path.join(path, "connections.csv")
    if not os.path.exists(conn_path):
        raise FileNotFoundError(f" can not in  {conn_path} find connections.csv")
    
    conn_df = pd.read_csv(conn_path)
    all_files = [f for f in os.listdir(path) if f.endswith('.csv') and f != 'connections.csv']
    tables: Dict[str, Table] = {}

    global_fkey_map = {}

    for file_name in all_files:
        table_key = file_name.replace(".csv", "")
        df = pd.read_csv(os.path.join(path, file_name))

        is_main = (table_key == task.entity_table)
        if is_main:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            if task.target_col in df.columns:
                if task.task_type in (
                    TaskType.BINARY_CLASSIFICATION,
                    TaskType.MULTICLASS_CLASSIFICATION,
                ):
                    df[task.target_col] = df[task.target_col].astype(int)

                    
                elif task.task_type == TaskType.REGRESSION:
                    df[task.target_col] = pd.to_numeric(
                        df[task.target_col], errors="raise"
                    ).astype(float)


        current_pkey = ''
        column_counts: Dict[str, int] = {}
        fkey_map: Dict[str, str] = {}

        out_conns = conn_df[conn_df["fk_table"] == file_name]
        for _, row in out_conns.iterrows():
            current_pkey = row["fk_pk"]
            fk_col = row["fk_column"]
            pk_table_name = str(row["pk_table"]).replace(".csv", "")

            if pd.isna(fk_col):
                continue
            if fk_col.strip() == current_pkey.strip():
                continue
            if fk_col not in df.columns:
                raise KeyError(f"Table {file_name} not contain fk_column '{fk_col}'")

            add_fkey_mapping(df, fkey_map, column_counts, fk_col, pk_table_name, global_fkey_map, table_key, path, file_name)


        in_conns = conn_df[conn_df["pk_table"] == file_name]

        for _, row in in_conns.iterrows():
            current_pkey = row["pk_pk"]
            pk_col = row["pk_column"]
            pk_pk = row["pk_pk"]
            fk_table_name = str(row["fk_table"]).replace(".csv", "")

            if pd.isna(pk_col) or pd.isna(pk_pk):
                continue

            if pk_col.strip() == pk_pk.strip():
                continue

            if pk_col not in df.columns:
                raise KeyError(f"Table {file_name} not contain pk_column '{pk_col}'")

            add_fkey_mapping(df, fkey_map, column_counts, pk_col, fk_table_name, global_fkey_map, table_key, path, file_name)
        if current_pkey not in df.columns:
            raise ValueError(f"Error：Table {file_name} not contain primary key '{current_pkey}'。")

        tables[table_key] = Table(
            df=df,
            fkey_col_to_pkey_table=fkey_map,
            pkey_col=current_pkey,
            is_need_edge=is_main
        )

    print(global_fkey_map)
    return Database(tables)

def add_fkey_mapping(
    df: pd.DataFrame,
    fkey_map: Dict[str, str],
    column_counts: Dict[str, int],
    local_col: str,
    target_table: str,
    global_fkey_map: Dict[str, int],
    table_key:str,
    path:str,
    file_name:str #df filename
) -> None:

    if local_col not in column_counts:
        column_counts[local_col] = 1
    else:
        column_counts[local_col] += 1

    if local_col in fkey_map and fkey_map[local_col] == target_table:
        return
    cols = df.columns.tolist()
    if local_col in fkey_map and local_col in cols:
        local_col_new = f'{local_col}_{target_table}'
        target_table_df = pd.read_csv(os.path.join(path, file_name))
        target_table_df[local_col_new] = target_table_df[local_col]
        target_table_df.to_csv(os.path.join(path, file_name), index=False)

        df[local_col_new] = df[local_col]

        global_fkey_map[local_col_new] = target_table
        fkey_map[local_col_new] = target_table
    else:
        global_fkey_map[local_col] = target_table
        fkey_map[local_col] = target_table
    # global_fkey_map[local_col] = target_table
    # fkey_map[local_col] = target_table


if __name__ == "__main__":
    from t4g.task.conn.conn_sped import ConnSpedClasTask
    from t4g.task.conn.conn_olist import ConnOlistClasTask

    task = ConnOlistClasTask
    create_database('/root/czq/code/sf-new_v5/raw_data/conn/olist/selected01', task)