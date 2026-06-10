from typing import Any, Dict, List, Optional,Tuple
from collections import defaultdict
import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet, TabTransformer
from t4g.encoder.encoder import MyTransformer
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv, GATConv, \
    GCNConv, GINConv, SGConv, GatedGraphConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import trim_to_layer
import torch.nn.functional as F
from torch.nn import ModuleDict, ModuleList


class HeteroEncoder(torch.nn.Module):
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
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroEncoder(torch.nn.Module):
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
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroGraph(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
            fusion: str = "mean",  # "mean" or "sum"
    ):
        print('new graph model! add residual')
        super().__init__()

        assert fusion in {"mean", "sum"}
        assert aggr in {"sum", "mean", "max", "min"}

        self.aggr = aggr
        self.fusion = fusion

        self.sage_convs = torch.nn.ModuleList()
        self.gat_convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            sage_conv = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (channels, channels),
                        channels,
                        aggr=aggr,
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.sage_convs.append(sage_conv)

            gat_conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (channels, channels),
                        channels,
                        heads=heads,
                        concat=False,
                        add_self_loops=False,
                        aggr=aggr,
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.gat_convs.append(gat_conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.sage_convs:
            conv.reset_parameters()

        for conv in self.gat_convs:
            conv.reset_parameters()

        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def _aggregate(self, xs):
        if len(xs) == 1:
            return xs[0]

        xs = torch.stack(xs, dim=0)

        if self.aggr == "sum":
            return xs.sum(dim=0)
        if self.aggr == "mean":
            return xs.mean(dim=0)
        if self.aggr == "max":
            return xs.max(dim=0)[0]
        if self.aggr == "min":
            return xs.min(dim=0)[0]

        raise ValueError(f"Unsupported aggr: {self.aggr}")

    def _fuse(self, sage_dict, gat_dict):
        out_dict = {}

        node_types = set(sage_dict.keys()) | set(gat_dict.keys())

        for node_type in node_types:
            sage_x = sage_dict.get(node_type, None)
            gat_x = gat_dict.get(node_type, None)

            if sage_x is None:
                out_dict[node_type] = gat_x
            elif gat_x is None:
                out_dict[node_type] = sage_x
            elif self.fusion == "sum":
                out_dict[node_type] = sage_x + gat_x
            elif self.fusion == "mean":
                out_dict[node_type] = 0.5 * (sage_x + gat_x)
            else:
                raise ValueError(f"Unsupported fusion: {self.fusion}")

        return out_dict

    def _sage_edge_weight(self, edge_index, num_dst_nodes):
        _, dst = edge_index

        if self.aggr == "sum":
            return torch.ones(
                edge_index.size(1),
                device=edge_index.device,
                dtype=torch.float,
            )

        if self.aggr == "mean":
            deg = torch.bincount(
                dst,
                minlength=num_dst_nodes,
            ).clamp(min=1)

            return 1.0 / deg[dst].float()
        return None

    def _sage_weights(self, x_dict, edge_index_dict):
        weights = {}

        for edge_type, edge_index in edge_index_dict.items():
            _, _, dst_type = edge_type

            if dst_type not in x_dict:
                continue

            weights[edge_type] = {
                "edge_index": edge_index,
                "weight": self._sage_edge_weight(
                    edge_index=edge_index,
                    num_dst_nodes=x_dict[dst_type].size(0),
                ),
                "weight_type": f"sage_{self.aggr}_structural",
            }

        return weights

    def _gat_forward_with_attention(self, gat_conv, x_dict, edge_index_dict):
        out_dict = {}
        attn_dict = {}

        for edge_type, conv in gat_conv.convs.items():
            if edge_type not in edge_index_dict:
                continue

            src_type, _, dst_type = edge_type

            if src_type not in x_dict or dst_type not in x_dict:
                continue

            edge_index = edge_index_dict[edge_type]

            if src_type == dst_type:
                x = x_dict[src_type]
            else:
                x = (x_dict[src_type], x_dict[dst_type])

            out, (edge_index_out, alpha) = conv(
                x,
                edge_index,
                return_attention_weights=True,
            )

            out_dict.setdefault(dst_type, []).append(out)

            attn_dict[edge_type] = {
                "edge_index": edge_index_out,
                "weight": alpha,  # [num_edges, heads]
                "weight_type": "gat_attention",
            }

        out_dict = {
            node_type: self._aggregate(xs)
            for node_type, xs in out_dict.items()
        }

        return out_dict, attn_dict

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
            return_attention_weights: bool = False,
    ):
        weights_dict = {}

        for layer_idx, (sage_conv, gat_conv, norm_dict) in enumerate(
                zip(self.sage_convs, self.gat_convs, self.norms)
        ):
            x_in_dict = x_dict

            sage_dict = sage_conv(x_in_dict, edge_index_dict)

            if return_attention_weights:
                gat_dict, gat_weights = self._gat_forward_with_attention(
                    gat_conv=gat_conv,
                    x_dict=x_in_dict,
                    edge_index_dict=edge_index_dict,
                )

                weights_dict[layer_idx] = {
                    "sage": self._sage_weights(
                        x_dict=x_in_dict,
                        edge_index_dict=edge_index_dict,
                    ),
                    "gat": gat_weights,
                }
            else:
                gat_dict = gat_conv(x_in_dict, edge_index_dict)

            x_msg_dict = self._fuse(sage_dict, gat_dict)

            x_out_dict = {}

            for key, x_prev in x_in_dict.items():
                x_msg = x_msg_dict.get(key, None)
                if x_msg is None:
                    x_out = x_prev
                else:
                    # residual connection
                    x_out = x_prev + x_msg

                x_out_dict[key] = norm_dict[key](x_out).relu()

            x_dict = x_out_dict

        if return_attention_weights:
            return x_dict, weights_dict

        return x_dict

#############################
# useless
#############################
class HeteroGraphNew(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            sage_conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            gat_conv = HeteroConv(
                {
                    edge_type: GATConv(
                        (channels, channels),
                        channels,
                        heads=heads,
                        concat=False,  
                        add_self_loops=False
                    )
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(ModuleDict({
                'sage': sage_conv,
                'gat': gat_conv
            }))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv_dict in self.convs:
            for conv in conv_dict.values(): 
                conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        attn_per_layer: List[Dict[EdgeType, Tensor]] = []

        for conv_dict, norm_dict in zip(self.convs, self.norms):
            sage_conv = conv_dict["sage"]
            gat_conv = conv_dict["gat"]

            x_sage = sage_conv(x_dict, edge_index_dict)

            x_gat: Dict[NodeType, Tensor] = {}
            layer_attn: Dict[EdgeType, Tensor] = {}

            for edge_type, conv in gat_conv.convs.items():
                src_type, _, dst_type = edge_type

                x_src = x_dict[src_type]
                x_dst = x_dict[dst_type]
                edge_index = edge_index_dict[edge_type]

                if src_type != dst_type:
                    input_pair = (x_src, x_dst)
                else:
                    input_pair = x_src

                out, (ei, attn_weights) = conv(input_pair, edge_index, return_attention_weights=True)

                if dst_type not in x_gat:
                    x_gat[dst_type] = out
                else:
                    x_gat[dst_type] += out

                layer_attn[edge_type] = attn_weights  

            x_dict = {
                key: x_sage[key] + x_gat[key]
                for key in x_sage.keys()
            }

            attn_per_layer.append(layer_attn)

            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict, attn_per_layer
    
class HeteroGraphGCN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GCNConv(channels, channels, add_self_loops=False)
                    # K is the number of propagation steps
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            conv = HeteroConv(
                {
                    edge_type: GATConv((channels, channels), channels, heads=heads, add_self_loops=False, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for step, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # x_dict = conv(x_dict, edge_index_dict)
            if step % 2 == 0:
                x_dict = conv(x_dict, edge_index_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphGIN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: GINConv(
                        torch.nn.Sequential(
                            torch.nn.Linear(channels, channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(channels, channels),
                        ), aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            conv = HeteroConv(
                {
                    edge_type: GATConv((channels, channels), channels, heads=heads, add_self_loops=False, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphGGNN(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        conv = HeteroConv(
            {
                edge_type: GatedGraphConv(channels, num_layers=num_layers, aggr=aggr)
                for edge_type in edge_types
            },
            aggr=aggr,
        )
        self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphNoWeight(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "sum",
            num_layers: int = 2,
            heads=1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr=aggr,
            )
            self.convs.append(conv)
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
            self,
            node_types: List[NodeType],
            edge_types: List[EdgeType],
            channels: int,
            aggr: str = "mean",
            num_layers: int = 1,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[NodeType, Tensor],
            num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
            num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Trim graph and features to only hold required data per layer:
            if num_sampled_nodes_dict is not None:
                assert num_sampled_edges_dict is not None
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                    num_sampled_edges_per_hop=num_sampled_edges_dict,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
