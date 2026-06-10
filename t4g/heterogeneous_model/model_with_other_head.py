from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType, EdgeType

from t4g.heterogeneous_model.nn import HeteroEncoder, HeteroGraphSAGE, HeteroGraph, HeteroGraphNoWeight, \
    HeteroGraphGCN, HeteroGraphGIN, HeteroGraphGGNN, HeteroGraphNew
from t4g.encoder.encoder import BaselineEncoder

import torch
from torch.nn import ModuleDict, Embedding, Linear, ReLU, Dropout, Sequential
from torch import Tensor
from typing import Dict, List, Any, Optional
import torch
from torch.nn import ModuleDict, Embedding, Linear, ReLU, Dropout, Sequential
from torch import Tensor
from typing import Dict, List, Any, Optional

class ModelW(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.gnn = HeteroGraph(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )
        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)


        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table])


import torch
from torch.nn import ModuleDict, Embedding, Linear, Sigmoid, Sequential
from torch import Tensor
from typing import Dict, List, Any, Optional

class GatedHeader(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gate = Sequential(
            Linear(in_channels, in_channels),
            Sigmoid()
        )
        self.out_lin = Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        gate_values = self.gate(x)
        return self.out_lin(x * gate_values)

class ModelW_Gated(torch.nn.Module):
    def __init__(
        self,
        data: Any,
        col_stats_dict: Dict[str, Dict[str, Dict[Any, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[str] = [],
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.gnn = HeteroGraph(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

        self.head = GatedHeader(
            in_channels=channels,
            out_channels=out_channels
        )

        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        
        for m in self.head.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: Any,
        entity_table: str,
    ) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table])


class ResHeader(torch.nn.Module):
    def __init__(self, channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.res_path = Sequential(
            Linear(channels, channels),
            ReLU(),
            Dropout(dropout),
            Linear(channels, channels)
        )
        self.out_lin = Linear(channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        h = x + self.res_path(x)
        return self.out_lin(h)

class ModelW_Res(torch.nn.Module):
    def __init__(
        self,
        data: Any,
        col_stats_dict: Dict[str, Dict[str, Dict[Any, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[str] = [],
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.gnn = HeteroGraph(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

        self.head = ResHeader(
            channels=channels,
            out_channels=out_channels,
            dropout=0.1
        )

        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        
        for m in self.head.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: Any,
        entity_table: str,
    ) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table])