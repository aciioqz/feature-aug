from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch_frame
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
from torch_frame.data.stats import StatType
from torch_geometric.data import Data
from torch_geometric.nn import LayerNorm, GATConv
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from t4g.encoder.encoder import MyTransformer
from t4g.model.graph import TableData



class ModelPool(torch.nn.Module):
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

        self.gnn = GATs(
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
        tf_dict = {}

        for node_type in data_dict:
            tf_value = data_dict[node_type].tf

            tf_dict[node_type] = tf_value

        x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict)

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
        x_dict,edge_index, weight = self.gnn(
            x_dict,
            data.edge_index,
        )

        pooled = global_mean_pool(x_dict, data.batch)

        head = self.head(pooled)

        return head, edge_index, weight

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

        self.gnn = GATs(
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
        tf_dict = {}

        for node_type in data_dict:
            tf_value = data_dict[node_type].tf

            tf_dict[node_type] = tf_value

        x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict)

        # x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict={
        #     node_type: data_dict[node_type].tf
        #     for node_type in data_dict
        # })

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
        x_dict,edge_index, weight = self.gnn(
            x_dict,
            data.edge_index,
        )
        head = self.head(x_dict)

        return head, edge_index, weight


class GATs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, aggr="mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        alphas = None
        for conv, norm in zip(self.convs, self.norms):
            x, (edge_index, alpha) = conv(x, edge_index, return_attention_weights=True)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return x, edge_index, alpha


class ModelDataWrapper:
    def __init__(self, data, data_dict, table_index, table_num, index_arr, loader, y, device, offset_arr):
        """
        data_dict = 'base': <t4g.model.graph.TableData object at 0x7fb6b11aa490>, 'kr_array2': <t4g.model.graph.TableData object at 0x7fb6b11aac10>, 'kr_array5': <t4g.model.graph.TableData object at 0x7fb6b11aadc0>,
        offset_arr = [1000, 2000, 3000]
        table_num = {0: 'base', 1: 'kr_array2', 2: 'kr_array5'}
        """
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
class Encoder(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=MyTransformer,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {}

        for node_type, tf in tf_dict.items():
            encoder = self.encoders[node_type]

            encoded_value = encoder(tf)

            x_dict[node_type] = encoded_value

        return x_dict

