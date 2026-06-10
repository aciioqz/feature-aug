import copy
import math
import os
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_frame.config import TextEmbedderConfig
from t4g.util.data_type import infer_df_stype
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

from t4g.data import BenchDataset
from t4g.datasets import get_dataset
from t4g.encoder.text_embedder import GloveTextEmbedding
from t4g.complete_graph.relation_batch.graph import make_complete_graph
from t4g.complete_graph.relation_batch.model_batch_no_encoder import Model, ModelDataWrapper
from t4g.util.file_utils import split_dataframe
from t4g.util.utils import plot_loss_with_acc


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels)
        self.attention_weights = None

    def forward(self, x, edge_index):
        x, attention_weights = self.gat1(x, edge_index, return_attention_weights=True)
        self.attention_weights = attention_weights
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        return x


class FeatureGraph:
    def __init__(self, row, feature_names):
        self.row = row
        self.feature_names = feature_names
        self.node_to_feature = {i: name for i, name in enumerate(feature_names)}
        self.feature_to_node = {name: i for i, name in enumerate(feature_names)}
        self.graph = None
        self.attention_weights = None

    def create_graph(self):
        n_features = len(self.feature_names)

        x = torch.tensor(self.row.values, dtype=torch.float).view(-1, 1)

        edge_index = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        self.graph = Data(x=x, edge_index=edge_index)
        return self.graph

    def set_attention_weights(self, attention_weights):
        self.attention_weights = attention_weights

    def get_feature_relationship(self):
        if self.attention_weights is None:
            return None

        relationships = {}
        edge_index = self.graph.edge_index.t().numpy()
        attention = self.attention_weights.detach().numpy()

        for (edge, weight) in zip(edge_index, attention):
            source = self.node_to_feature[edge[0]]
            target = self.node_to_feature[edge[1]]
            relationships[(source, target)] = weight

        return relationships

    def visualize_relationship(self):
        if self.attention_weights is None:
            return

        G = nx.DiGraph()

        for feature in self.feature_names:
            G.add_node(feature)

        relationships = self.get_feature_relationship()
        for (source, target), weight in relationships.items():
            G.add_edge(source, target, weight=weight)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=1000)
        nx.draw_networkx_labels(G, pos)

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights)

        plt.title("Feature Relationship Graph")
        plt.show()

class Args:
    def __init__(self, lr=0.005, epochs=300, dataset=None, task=None, batch_size=128, layers=2):
        self.dataset = dataset
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.channels = 128
        self.aggr = "sum"
        self.num_neighbors = 128
        self.num_workers = 0
        self.layers = layers


def main(args: Args):
    seed_value =  42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_length = 100
    column_num = 20
    root_path = '/root/autodl-tmp/czq/code/sf-new/raw_data/fraud/feature'
    base_path = os.path.join(root_path, 'train_transaction_sample1w.csv')
    base_column = pd.read_csv(base_path, nrows=total_length).columns
    df = pd.read_csv(base_path, usecols=base_column[:column_num], nrows=total_length)

    df.insert(0, 'id', range(len(df)))
    id_column = 'id'
    label_column = 'isFraud'
    df_y = df[[id_column, label_column]]

    df = df.drop(columns=['TransactionID'])

    table_name_prefix = 'table'
    subtable_dict: Dict[str, pd.DataFrame] = split_dataframe(table_name_prefix, df, [id_column, label_column])
    feature_names = df.columns.tolist()

    col_to_stype = infer_df_stype(df)

    data: Data
    data, data_dict = make_complete_graph(
        tables=subtable_dict,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=device), batch_size=256),
        pk_column_name=id_column,
        label_column_name=label_column,
    )
    batch_size = 128
    single_subgraph_size = len(subtable_dict)
    length_1 = int(total_length * 0.6)
    length_2 = int(total_length * 0.2)


    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True



    train_index_arr, train_y = get_id_and_label(df_y=df_y, start_l=0, end_l=length_1, table_num=single_subgraph_size)
    val_index_arr, val_y = get_id_and_label(df_y=df_y, start_l=length_1,  end_l=length_1 + length_2, table_num=single_subgraph_size)
    test_index_arr, test_y = get_id_and_label(df_y=df_y, start_l=length_1 + length_2,  end_l=total_length, table_num=single_subgraph_size)


    loader_dict: Dict[str, NeighborLoader] = {}
    loader_dict["train"] = NeighborLoader(data, input_nodes=torch.from_numpy(train_index_arr),
                                          batch_size=args.batch_size, shuffle=True, num_neighbors=[-1, -1])
    loader_dict["val"] = NeighborLoader(data, input_nodes=torch.from_numpy(val_index_arr),
                                        batch_size=args.batch_size, shuffle=False, num_neighbors=[-1, -1])
    loader_dict["test"] = NeighborLoader(data, input_nodes=torch.from_numpy(test_index_arr),
                                         batch_size=args.batch_size, shuffle=False, num_neighbors=[-1, -1])

    model = Model(data_dict, args, out_channels, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset: BenchDataset = get_dataset(name='base-fraud', process=True, path=root_path)
    task = dataset.get_task('fraud-classification')

    loss_values = []
    acc_values = []
    state_dict = None
    best_val_metric = 0 if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        print(f"seed: {seed_value}, Epoch: {epoch:02d}")
        # data, data_dict, loader, y, device
        train_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict,
                                              loader=loader_dict["train"],
                                              y=train_y, device=device)
        train_loss = train_batch(model, optimizer, loss_fn, train_data_wrapper)
        loss_values.append(train_loss)
        val_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict, loader=loader_dict["val"],
                                            y=val_y, device=device,)
        val_pred = test_batch(model, val_data_wrapper, args.batch_size)
        val_metrics = task.evaluate(val_pred, val_y)
        acc_values.append(val_metrics["accuracy"])

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
        # torch.save(model, f'./model_seed_value_{seed_value}_epoch_{epoch}.pth')
    model.load_state_dict(state_dict)
    val_pred = test_batch(model, val_data_wrapper, args.batch_size)
    val_metrics = task.evaluate(val_pred, val_y)
    print(f"Best Val metrics: {val_metrics}")
    plot_loss_with_acc(loss_values, acc_values)

    test_data_wrapper = ModelDataWrapper(data=data, data_dict=data_dict, loader=loader_dict["test"],
                                         y=test_y, device=device)
    test_pred = test_batch(model, test_data_wrapper, args.batch_size)
    test_metrics = task.evaluate(test_pred, test_y)
    print(f"Best test metrics: {test_metrics}")

    result = {
        'lr': args.lr,
        'epochs': args.epochs,
        'seed': seed_value,
        'average_precision': test_metrics['average_precision'],  
        'accuracy': test_metrics['accuracy'],  
        'f1': test_metrics['f1'],  
        'roc_auc': test_metrics['roc_auc']  
    }
    return result


def get_id_and_label(df_y:pd.DataFrame, start_l, end_l, table_num):
    x = np.arange(0, end_l)
    id_list = x * table_num
    # df_y has two column [id, label]
    label_value = df_y.iloc[start_l:end_l, 1]
    return id_list, label_value

@torch.no_grad()
def test_batch(model, dataWrapper: ModelDataWrapper, batch_size) -> np.ndarray:
    model.eval()
    pred_list = []
    y_list_index = []
    for batch in dataWrapper.loader:
        batch = batch.to(dataWrapper.device)
        print(batch)
    

def train_batch(model, optimizer, loss_fn, dataWrapper):
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = len(dataWrapper.loader)
    for batch in dataWrapper.loader:
        batch = batch.to(dataWrapper.device)
        print(batch)
    


