from torch_frame.data import Dataset, TensorFrame
import torch
import numpy as np
from torch_frame import Metric, TaskType, TensorFrame, stype
from torch import Tensor
from torch_frame.nn import LinearEncoder, EmbeddingEncoder, MultiCategoricalEmbeddingEncoder, LinearEmbeddingEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

def table_encoder(dataset: Dataset):
    print("table_encoder")
    frame = dataset.tensor_frame
    feat_dict = frame.feat_dict

    first_dim_size = 0
    for key, value in feat_dict.items():
        if 'embedding' in str(key):
            first_dim_size = value.values.shape[0]
        else:
            first_dim_size = value.shape[0]
        if first_dim_size > 0:
            break

    start_index = 0
    res_dict = {}
    for i in range(first_dim_size):
        values = []
        for key, value in feat_dict.items():
            if 'embedding' in str(key):
                values.append(value.values[i])
            else:
                j_temp = value[i].view(-1)
                values.append(j_temp)

        concatenated_tensor = torch.cat(values, dim=0)

        res_dict[i] = concatenated_tensor

    return res_dict


def table_encoder_list(dataset: Dataset):
    print("table_encoder_list")
    frame = dataset.tensor_frame
    feat_dict = frame.feat_dict

    first_dim_size = 0
    for key, value in feat_dict.items():
        if 'embedding' in str(key):
            first_dim_size = value.values.shape[0]
        else:
            first_dim_size = value.shape[0]
        if first_dim_size > 0:
            break

    start_index = 0
    res_dict = []
    for i in range(first_dim_size):
        values = []
        for key, value in feat_dict.items():
            if 'embedding' in str(key):
                values.append(value.values[i])
            else:
                j_temp = value[i].view(-1)
                values.append(j_temp)

        concatenated_tensor = torch.cat(values, dim=0)

        res_dict.append(concatenated_tensor)

    return res_dict


def convert_input(
    dataset: Dataset,
):
# ) -> list[Tensor]:
    tf = dataset.tensor_frame
    tf = tf.cpu()

    feats: list[Tensor] = []
    if stype.categorical in tf.feat_dict:

        stats_list = [
            dataset.col_stats[col_name]
            for col_name in tf.col_names_dict[stype.categorical]
        ]
        feat_cat = tf.feat_dict[stype.categorical].clone()
        col_names = tf.col_names_dict[stype.categorical]
        encoder = EmbeddingEncoder(
            out_channels = 8,
            stats_list=stats_list,
            stype=stype.categorical,
        )
        #  x shape is (len, column_number, out_channels)
        x = encoder(feat_cat, col_names)
        tensor_squeezed = tensor3d_2_2d(x)
        feats.append(tensor_squeezed)


    if stype.numerical in tf.feat_dict:
        stats_list = [
            dataset.col_stats[col_name]
            for col_name in tf.col_names_dict[stype.numerical]
        ]
        feat_cat = tf.feat_dict[stype.numerical].clone()
        col_names = tf.col_names_dict[stype.numerical]
        encoder = LinearEncoder(
            8,
            stats_list=stats_list,
            stype=stype.numerical,
        )
        x = encoder(feat_cat, col_names)
        tensor_squeezed = tensor3d_2_2d(x)
        feats.append(tensor_squeezed)

    if stype.embedding in tf.feat_dict:
        encoder = LinearEmbeddingEncoder
        embedding = encoder(tf.feat_dict[stype.embedding])

        stats_list = [
            dataset.col_stats[col_name]
            for col_name in tf.col_names_dict[stype.embedding]
        ]
        feat_cat = tf.feat_dict[stype.embedding].clone()
        col_names = tf.col_names_dict[stype.embedding]
        encoder = LinearEmbeddingEncoder(
            out_channels=100,
            stats_list=stats_list,
            stype=stype.embedding,
        )
        x = encoder(feat_cat, col_names)
        tensor_squeezed = tensor3d_2_2d(x)
        feats.append(tensor_squeezed)
    if len(feats) == 0:
        raise ValueError("The input TensorFrame object is empty.")
    feat = torch.cat(feats, dim=-1)
    return feat


def tensor3d_2_2d(x):
    """
    x shape is (len, column_number, out_channels), for example convert x=tensor(1000,3,8) into x=tensor(1000,24)
    """
    shape = x.size()

    new_shape = shape[0], shape[1] * shape[2]

    output = x.view(new_shape)  

    print(output.shape)  
    return output


def nearest_neighbors_cosine_similarity(samples, top_k=1, threshold=0.9):
    if samples.requires_grad:
        samples = samples.detach()
    norms = torch.linalg.norm(samples, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    samples_normalized = samples / norms

    neigh = NearestNeighbors(n_neighbors=top_k + 1, metric='cosine')
    neigh.fit(samples_normalized)
    distances, indices = neigh.kneighbors(samples_normalized)

    high_similarity_pairs = []

    for i in range(indices.shape[0]):
        for j in range(1, top_k + 1):
            sim = 1 - distances[i, j]
            if sim > threshold:
                high_similarity_pairs.append([i, indices[i, j]])

    high_similarity_pairs = np.array(high_similarity_pairs).T
    return torch.tensor(high_similarity_pairs)



def sampled_nearest_neighbors_cosine_similarity(samples, top_k=1, threshold=0.9):
    """
    random simple 1w 
    """
    n_total = samples.shape[0]
    
    if n_total > 10000:
        perm = torch.randperm(n_total)[:10000]
        subset = samples[perm]
    else:
        subset = samples
        perm = torch.arange(n_total)

    if subset.requires_grad:
        subset = subset.detach()
        
    norms = torch.linalg.norm(subset, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    subset_norm = subset / norms


    sim_matrix = torch.mm(subset_norm, subset_norm.t())

    sim_matrix.fill_diagonal_(-1.0)

    top_vals, top_idxs = torch.topk(sim_matrix, k=top_k, dim=1)

    mask = top_vals > threshold
    
    row_indices = torch.arange(subset_norm.shape[0]).view(-1, 1).expand_as(top_idxs)
    
    high_sim_subset_indices = torch.stack([row_indices[mask], top_idxs[mask]], dim=0)

    real_indices = perm[high_sim_subset_indices]
    
    return real_indices



def compute_cosine_similarity_matrix(samples, top_k=1000):
    
    if samples.requires_grad:
        samples = samples.detach()
    similarity_matrix = cosine_similarity(np.array(samples))
    np.fill_diagonal(similarity_matrix, 0)

    n_samples = similarity_matrix.shape[0]
    high_similarity_pairs = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            high_similarity_pairs.append((similarity_matrix[i, j], i, j))

    high_similarity_pairs = sorted(high_similarity_pairs, reverse=True, key=lambda x: x[0])
    top_k_pairs = high_similarity_pairs[:top_k]

    pair_indices = torch.tensor([[pair[1], pair[2]] for pair in top_k_pairs]).t()

    return pair_indices

