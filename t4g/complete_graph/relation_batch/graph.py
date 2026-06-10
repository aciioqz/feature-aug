from typing import Dict, NamedTuple, Optional

import pandas as pd
import torch
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_geometric.data import Data
from torch_frame import DataFrame, Metric, TaskType, TensorFrame, stype

from t4g.data import Table
from t4g.data.task import Task
from t4g.util.file_utils import get_stype_proposal


def to_unix_time(ser: pd.Series) -> Tensor:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp
    (in seconds)."""
    return torch.from_numpy(ser.astype(int).values) // 10 ** 9




def make_complete_graph(
        tables: Dict[str, pd.DataFrame],
        text_embedder_cfg: Optional[TextEmbedderConfig] = None,
        pk_column_name:str=None,
        label_column_name: str=None,
):
    """
    Each table shares the same PK and LABEL column.
    """
    print("start construct complete graph")

    data = Data()
    data_dict = {}
    nodes = []
    edge_index = []
    x = []
    y = []
    tensor_list = []
    # if cache_dir is not None:
    #     os.makedirs(cache_dir, exist_ok=True)

    # Find the maximum id in all tables to know the range we need to iterate over
    max_id = next(iter(tables.values()))['id'].max()

    for i in range(max_id + 1):
        current_node_indices = []

        for name, df in tables.items():

            node = df[df[pk_column_name] == i]
            if not node.empty:
                node_index = len(nodes)

                nodes.append(i)
                current_node_indices.append(node_index)
                stype_proposal, no_list = get_stype_proposal(df, [pk_column_name, label_column_name])
                dataset = Dataset(
                    df=df,
                    col_to_stype=stype_proposal,
                    col_to_text_embedder_cfg=text_embedder_cfg,
                ).materialize()
                if name not in data_dict:
                    data_dict[name] = TableData()
                data_dict[name].tf = dataset.tensor_frame
                data_dict[name].col_stats = dataset.col_stats

                x.append(node_index)
                y.append(node.iloc[0][1])

        for j in current_node_indices:
            for k in current_node_indices:
                if j != k:
                    edge_index.append(torch.tensor([[j], [k]], dtype=torch.long))

    print('add_edge_end')
    all_edge = torch.cat(edge_index, dim=1)
    data.edge_index = all_edge
    data.x = torch.tensor(x)
    data.y = torch.tensor(y)
    data.validate()

    return data, data_dict


    return table_index, table_num, offset_arr


def get_embedding(i:int=0, tf: TensorFrame=None) :
    if stype.categorical in tf.feat_dict:
        feat = tf.feat_dict[stype.categorical].numpy()
        embedding = feat[i]
        return embedding

    elif stype.numerical in tf.feat_dict:
        feat = tf.feat_dict[stype.numerical].numpy()
        embedding = feat[i]
        return embedding

    elif stype.embedding in tf.feat_dict:
        feat = tf.feat_dict[stype.embedding]
        feat = feat.values
        feat = feat.view(feat.size(0), -1).numpy()
        embedding = feat[i]
        return embedding






class AttachTargetTransform:
    r"""Adds the target label to the heterogeneous mini-batch.
    The batch consists of disjoins subgraphs loaded via temporal sampling.
    The same input node can occur twice with different timestamps, and thus
    different subgraphs and labels. Hence labels cannot be stored in the graph
    object directly, and must be attached to the batch after the batch is
    created."""

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: Data) -> Data:
        batch.y = self.target
        return batch


class TrainTableInput(NamedTuple):
    nodes: Tensor
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]
    entity_table: str


def get_train_table_input(
        table: Table,
        task: Task,
) -> TrainTableInput:
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = to_unix_time(table.df[table.time_col])

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == "multiclass_classification":
            target_type = int
        target = torch.from_numpy(table.df[task.target_col].values.astype(target_type))
        transform = AttachTargetTransform(task.entity_table, target)

    return TrainTableInput(
        nodes=nodes,
        time=time,
        target=target,
        transform=transform,
        entity_table=task.entity_table,
    )


class TableData:
    def __init__(self):
        self.tf = None
        self.col_stats = None
        self.time = None


