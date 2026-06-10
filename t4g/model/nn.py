from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import LayerNorm, SAGEConv, GATConv, GCNConv, GCN2Conv
from torch_geometric.typing import NodeType
from torch.nn import LayerNorm, Dropout, ModuleList

class Encoder(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
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
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, aggr="mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(in_channels, hidden_channels, aggr=aggr)
            self.convs.append(conv)
            self.convs.append(GATConv(in_channels, hidden_channels, aggr=aggr))
            in_channels = hidden_channels
        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x

class GATs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, aggr="mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(in_channels, hidden_channels, heads=1)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return x

class GraphSAGEDROP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, aggr="mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(in_channels, hidden_channels, aggr=aggr)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(0.2)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, aggr="mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(in_channels, hidden_channels, aggr=aggr)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x


class GraphSAGE_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, aggr="mean"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        sage_layer1 = SAGEConv(in_channels, hidden_channels, aggr = aggr)
        self.convs.append(sage_layer1)
        sage_layer2 = SAGEConv(hidden_channels, hidden_channels, aggr = aggr)
        self.convs.append(sage_layer2)
        gat_layer = GATConv(hidden_channels, hidden_channels, heads = 1, aggr = aggr)
        self.convs.append(gat_layer)

        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x


class GCNII(torch.nn.Module):
    def __init__(self, embedding_dim_in, embedding_dim_out, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = nn.Linear(embedding_dim_in, embedding_dim_out)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(embedding_dim_out, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins(x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)

        return x

