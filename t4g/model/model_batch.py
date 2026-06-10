from typing import Dict

import torch
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from t4g.model.graph import TableData
from t4g.model.nn import Encoder, GraphSAGE


class Model(torch.nn.Module):
    def __init__(self, data_dict, args, out_channels, layers):
        super().__init__()

        self.encoder = Encoder(
            channels=args.channels,
            node_to_col_names_dict={
                node_type: data_dict[node_type].tf.col_names_dict
                for node_type in data_dict
            },
            node_to_col_stats={
                node_type: data_dict[node_type].col_stats
                for node_type in data_dict
            },
        )

        self.gnn = GraphSAGE(
            in_channels=args.channels,
            hidden_channels=args.channels,
            aggr=args.aggr,
            num_layers=layers
        )
        self.head = MLP(
            args.channels,
            out_channels=out_channels,
            num_layers=1,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
            self,
            data: Data,
            data_dict,
            table_num,
            index_dict,
            inverse_index_dict,
    ) -> Tensor:
        x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict={
            node_type: data_dict[node_type].tf
            for node_type in data_dict
        })

        tensors_to_concat = []

        data_x = data.x.tolist()
        for b in data_x:
            table_index = inverse_index_dict[b]
            table_name = table_num[table_index]
            tuple_index = index_dict[table_index][b]
            temp_tuple_tensor = x_dict[table_name][tuple_index]
            tensors_to_concat.append(temp_tuple_tensor)
            # tensors_to_concat = torch.cat((tensors_to_concat, temp_tuple_tensor), dim=0)
        tensors_to_concat = torch.stack(tensors_to_concat, dim=0)
        x_dict = tensors_to_concat
        x_dict = self.gnn(
            x_dict,
            data.edge_index,
        )
        head = self.head(x_dict)

        return head


class ModelDataWrapper:
    def __init__(self, data, data_dict, table_index, table_num, index_arr, loader, y, device, offset_arr):

        self.data = data
        self.data_dict = data_dict
        self.table_index = table_index
        self.table_num = table_num
        self.index_arr = index_arr
        self.loader = loader
        self.y = y
        self.device = device
        self.offset_arr = offset_arr


def get_batch_data_dict(batch, dataWrapper):
    index_dict, inverse_index_dict = find_indices(dataWrapper.offset_arr, batch.x.tolist())
    data_dict = {}
    for key, value in index_dict.items():
        offset = dataWrapper.offset_arr[key]
        table_name = dataWrapper.table_num[key]
        
        tuple_id_dict = dataWrapper.table_index[table_name]
        inverted_tuple_dict = {value: key for key, value in tuple_id_dict.items()}
        id_list = []
        for v in value:
            id_list.append(inverted_tuple_dict[v])
        table_data = dataWrapper.data_dict[table_name]

        data_dict[table_name] = TableData()
        new_tf = extract_rows(table_data, id_list)
        data_dict[table_name].tf = new_tf.to(dataWrapper.device)
        data_dict[table_name].col_stats = table_data.col_stats

    return data_dict, index_dict, inverse_index_dict


def extract_rows(data: TableData, indices) -> TensorFrame:

    new_feat_dict = {}

    for key, tensor in data.tf.feat_dict.items():
        if isinstance(tensor, torch.Tensor):
            new_feat_dict[key] = tensor[indices]
        elif isinstance(tensor, MultiEmbeddingTensor):
            new_values = tensor.values[indices, :]
            new_num_rows = len(indices)
            new_feat_dict[key] = MultiEmbeddingTensor(
                num_rows=new_num_rows,
                num_cols=tensor.num_cols,
                values=new_values,
                offset=tensor.offset
            )
        else:
            raise ValueError(f'Unsupported tensor type: {type(tensor)} for key: {key}')

    new_tf = TensorFrame(
        feat_dict=new_feat_dict,
        col_names_dict=data.tf.col_names_dict,
        y=data.tf.y[indices] if data.tf.y is not None else None
    )

    return new_tf


def find_indices(arr, batch_x):
    result_dict = {}
    inverse_dict = {}
    for b in batch_x:
        for i, a in enumerate(arr):
            if b < a:
                if i not in result_dict:
                    result_dict[i] = []
                result_dict[i].append(b)
                inverse_dict[b] = i
                break
    return result_dict, inverse_dict



