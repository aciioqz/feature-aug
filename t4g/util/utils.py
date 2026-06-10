import shutil
from pathlib import Path
from typing import Dict
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pooch
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import numpy as np


def to_unix_time(column: pd.Series) -> pd.Series:
    times = pd.to_datetime(column, errors='coerce')
    """convert a timestamp column to unix time"""
    result = pd.to_datetime(column).astype('int64') // 10**9 // 3600
    result[times.isnull()] = 0
    return result
    # return pd.to_datetime(column).astype("datetime64[s]")


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    shutil.unpack_archive(zip_path, unzip_path)
    return unzip_path


def get_df_in_window(df, time_col, row, delta):
    return df[
        (df[time_col] > row["timestamp"]) & (df[time_col] <= (row["timestamp"] + delta))
        ]


def read_csv_to_dict(path: str, key_column: str, value_column: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    values = df[value_column].values
    keys = df[key_column]
    if len(values) != len(keys):
        raise ValueError("Array length mismatch.")

    my_dict = {key: value for key, value in zip(keys, values)}

    return my_dict


import torch


def find_k_nearest_neighbors(m, k):
    num_tensors = len(m)
    neighbors_list = []  
    dists = torch.cdist(m, m, p=2)
    for i in range(num_tensors):
        curr_dists = dists[i]
        curr_dists[i] = float('inf')
        similar_values, topk_indices = curr_dists.topk(k, largest=False)
        similarities = 1 / (1 + similar_values.float())
        value_list = similarities.tolist()
        index_list = topk_indices.tolist()
        for index, value in zip(index_list, value_list):
            neighbors_list.append([i] + [index] + [round(value, 5)])

    return neighbors_list


def find_k_nearest_neighbors_list(tensor, k=1, threshold=0.9):
    num_tensors = len(tensor)
    neighbors_list = []  
    similar_list = []
    dists = torch.cdist(tensor, tensor, p=2)
    for i in range(num_tensors):
        curr_dists = dists[i]
        curr_dists[i] = float('inf')
        similar_values, topk_indices = curr_dists.topk(k, largest=False)
        similarities = 1 / (1 + similar_values.float())
        value_list = similarities.tolist()
        index_list = topk_indices.tolist()
        for index, value in zip(index_list, value_list):
            if value >= 0.99:
                neighbors_list.append([i] + [index])
                if value == 1:
                    similar_list.append(0.99999)
                else:
                    similar_list.append(round(value, 5))

    print("add knn edge size:" + str(len(similar_list)))

    return neighbors_list, similar_list


def find_top_similar_rows(tensor, k=1, threshold=0.999):
    has_nan = torch.isnan(tensor).any()

    if has_nan or has_nan:
        tensor = torch.nan_to_num(tensor, nan=0.0)
        print("The tensor contains null values.")
    else:
        print("The tensor contains no null values.")
    neighbors_list = []  
    similar_list = []
  
    if not tensor.is_floating_point():
        tensor = tensor.float()
    similarity_matrix = cosine_similarity_matrix(tensor)

    mask = torch.eye(tensor.shape[0], device=tensor.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))

    top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    high = top_k_similarities[top_k_similarities >= threshold].tolist()
   
    while True:
        top_k_similarities, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        high = top_k_similarities[top_k_similarities >= threshold].tolist()
        if len(high) < similarity_matrix.shape[0]:
            break
        if threshold >= 0.99999:
            threshold = 0.99999
            break
        elif threshold >= 0.9999:
            threshold += 0.00001
        elif threshold >= 0.999:
            threshold += 0.0001
        elif threshold >= 0.99:
            k=1
            threshold += 0.001
        elif threshold >= 0.9:
            threshold += 0.01
        else:
            threshold += 0.1

    results = []
    for i in range(tensor.shape[0]):
        row_similarities = top_k_similarities[i]
        row_indices = top_k_indices[i]

        valid_mask = row_similarities >= threshold
        valid_similarities = row_similarities[valid_mask].tolist()
        valid_indices = row_indices[valid_mask].tolist()

        if len(valid_indices) > 0:
            for index, value in zip(valid_indices, valid_similarities):
                if value >= threshold:
                    if not neighbors_list.__contains__([i] + [index]):
                        neighbors_list.append([i] + [index])
                        if value == 1:
                            value = 0.99999
                        else:
                            value = round(value, 5)
                        similar_list.append(value)
                        if not neighbors_list.__contains__([i] + [index]):
                            neighbors_list.append([index] + [i])
                            similar_list.append(value)

    print("add knn edge size:" + str(len(similar_list)))


    return neighbors_list, similar_list

def cosine_similarity_matrix(tensor):
    normalized = F.normalize(tensor, p=2, dim=1)
    similarity = torch.mm(normalized, normalized.t())
    return similarity

def euclidean_distance_matrix(tensor):
    dist_matrix = torch.cdist(tensor, tensor, p=2)
    return dist_matrix

def draw(graph):
    nids = graph.n_id
    graph = to_networkx(graph)
    for i, nid in enumerate(nids):
        graph.nodes[i]['txt'] = str(nid.item())
    node_labels = nx.get_node_attributes(graph, 'txt')

    nx.draw_networkx(graph, labels=node_labels, node_color='#00BFFF')
    plt.axis("off")
    plt.show()


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)  
    ax1.plot(range(len(loss_history)), loss_history,
             c=(np.array([255, 71, 90]) / 255.))  
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


