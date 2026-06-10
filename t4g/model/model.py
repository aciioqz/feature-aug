from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
import numpy as np
from t4g.model.nn import Encoder, GraphSAGE


class Model(torch.nn.Module):
    def __init__(self, data_dict, args, out_channels):
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
            table_num
    ) -> Tensor:
        x_dict: Dict[NodeType, Tensor] = self.encoder(tf_dict={
            node_type: data_dict[node_type].tf
            for node_type in data_dict
        })
        tensors_to_concat = []
        for key in sorted(table_num.keys()):
            cur_table_name = table_num[key]
            if cur_table_name in x_dict:
                tensors_to_concat.append(x_dict[cur_table_name])

        if tensors_to_concat:
            x_dict = torch.cat(tensors_to_concat, dim=0)
        else:
            print("No tensors available for concatenation.")

        x_dict = self.gnn(
            x_dict,
            data.edge_index,
        )
        head = self.head(x_dict)

        return head


class ModelDataWrapper:
    def __init__(self, data, data_dict, table_index, table_num, index_arr):
        self.data = data
        self.data_dict = data_dict
        self.table_index = table_index
        self.table_num = table_num
        self.index_arr = index_arr


@torch.no_grad()
def test(model, dataWrapper: ModelDataWrapper) -> np.ndarray:
    model.eval()
    pred_list = []
    pred = model(
        dataWrapper.data,
        dataWrapper.data_dict,
        dataWrapper.table_num
    )
    pred_raw = pred.view(-1) if pred.size(1) == 1 else pred
    pred = pred_raw[dataWrapper.index_arr]
    return pred


def train(model, optimizer, loss_fn, train_y, device, dataWrapper: ModelDataWrapper) -> Dict[str, float]:
    model.train()
    loss_accum = count_accum = 0
    optimizer.zero_grad()
    pred = model(
        dataWrapper. data,
        dataWrapper. data_dict,
        dataWrapper. table_num
    )
    pred_raw = pred.view(-1) if pred.size(1) == 1 else pred
    pred = pred_raw[dataWrapper. train_index_arr]
    loss = loss_fn(pred, torch.tensor(train_y, dtype=torch.float32).to(device))
    loss.backward()
    optimizer.step()
    loss_accum += loss.detach().item() * pred.size(0)
    count_accum += pred.size(0)
    l = loss_accum / count_accum
    # print(f"loss: {l}")
    return l, pred



