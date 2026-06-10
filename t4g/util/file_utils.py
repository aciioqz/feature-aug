from typing import Any, Dict, List

import pandas as pd
from torch_frame.utils import infer_series_stype
import random

def get_stype_proposal(df: pd.DataFrame, keys: List[str]) -> Dict[str, Any]:
    r"""Propose stype for columns of a table
    """
    inferred_col_to_stype_dict = {}
    none_type_columns = []
    for column in df.columns:
        if column not in keys:
            inferred_type = infer_series_stype(df[column])
            if inferred_type is None:
                none_type_columns.append(column)
            else:
                inferred_col_to_stype_dict[column] = inferred_type

    return inferred_col_to_stype_dict, none_type_columns


def split_and_save_csv(table_name: str, df: pd.DataFrame, exclude_columns: list):
    """
    Splits a CSV file into multiple smaller CSV files based on inferred column data types.

    :param file_path: Path to the input CSV file
    :param exclude_columns: List of column names to be excluded from the type inference
    :param output_folder: Folder where the output CSV files will be saved
    """
    column_types, none_type_columns = get_stype_proposal(df, exclude_columns)

    # Creating a dictionary to store DataFrames for each data type
    dataframes = {}

    result = {}
    # Separating columns by data type
    for column, dtype in column_types.items():
        if dtype not in dataframes:
            dataframes[dtype] = pd.DataFrame()
        dataframes[dtype][column] = df[column]

    for dtype, dataframe in dataframes.items():
        for column in exclude_columns:
            if column in df.columns:
                dataframe[column] = df[column]

    # Adding columns with None type to their own DataFrame
    if none_type_columns:
        if 'none' not in dataframes:
            dataframes['none'] = pd.DataFrame()
        for column in none_type_columns:
            dataframes['none'][column] = df[column]

    # Saving each DataFrame to a separate CSV file
    i = 0
    for dtype, dataframe in dataframes.items():
        temp_name = f"{table_name}_{i}"
        result[temp_name] = dataframe
        i = i + 1

    print("CSV files have been splited successfully.")
    return result


def split_dataframe(table_name: str, df: pd.DataFrame, keys: List[str]) -> Dict[str, pd.DataFrame]:
    
    columns = df.columns.tolist()
    print(f'columns:{columns}---key:{keys}')

    features = [col for col in columns if col not in keys]
    sub_tables = {}

    for index, feature in enumerate(features):
        if feature in keys:
            continue
        new_df = df[keys + [feature]]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
    print(f"split candidate table: {table_name} end！")
    return sub_tables
    

def split_dataframe_random(table_name: str, df: pd.DataFrame, keys: List[str]) -> Dict[str, pd.DataFrame]:
    

    columns = df.columns.tolist()

    features = [col for col in columns if col not in keys]
    random.shuffle(features)
    sub_tables = {}
    groups = []

    while len(features) >= 2:
        group = random.sample(features, 2)
        groups.append(group)

        for feature in group:
            features.remove(feature)
    if features:
        groups.append(features)


    for index, feature in enumerate(groups):
        if feature in keys:
            continue
        new_df = df[keys + feature]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
    return sub_tables

def split_dataframe_relation(table_name: str, df: pd.DataFrame, keys: List[str], relation_list:List[str]) \
        -> Dict[str, pd.DataFrame]:

    """
    df (pd.DataFrame): DataFrame。
    keys (list): [id, label]
    List[str]: ['Name-Pclass']
    """


    columns = df.columns.tolist()

    relation_list = [item.split('-') for item in relation_list]
    relation_set = set(item for sublist in relation_list for item in sublist)
    combined_set = relation_set.union(keys)

    features = [col for col in columns if col not in combined_set]
    sub_tables = {}

    index = 0
    for feature in relation_list:
        new_df = df[keys + feature]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
        index = index + 1

    for feature in features:
        new_df = df[keys + [feature]]
        new_file_name = f"{table_name}_{index}"
        sub_tables[new_file_name] = new_df
        index = index + 1
    return sub_tables

def add_foreign_key_columns(tables, id_column):
    all_fk_table = {}
    all_table_names = list(tables.keys())

    for target_table_name, target_df in tables.items():
        fk_table = {}
        for other_table_name in all_table_names:
            if other_table_name != target_table_name:
                other_ids = tables[other_table_name][id_column]
               
                target_df.loc[:, other_table_name + '_id'] = other_ids.values
                fk_table[other_table_name + '_id'] = other_table_name
        all_fk_table[target_table_name] = fk_table
        tables[target_table_name] = target_df

    return tables, all_fk_table


