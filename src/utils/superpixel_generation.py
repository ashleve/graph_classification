import multiprocessing
import os
import time
from itertools import repeat

import networkx as nx
import numpy as np
import torch
from skimage.segmentation import slic
from torch_geometric.data import Data

# tqdm for progress bar
from tqdm.auto import tqdm

# apply patch to enable progress bar with multiprocessing,
# requires python 3.8+
# see https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
# from multiprocessing_istarmap import multiprocessing_istarmap
from src.utils.multiprocessing_istarmap import multiprocessing_istarmap


def convert_img_dataset_to_superpixel_graph_dataset(
    images, labels, desired_nodes: int = 75, num_workers: int = 4, **slic_kwargs
):
    """Convert torchvision dataset to superpixel graphs

    Args:
        dataset (Tuple): Tuple of img array and label array
        desired_nodes (int):  Desired number of superpixels per img.
            The actual number is usually a bit different due to the
            way SLIC algorithm behaves.
        num_workers (int): Number of processes for dataset conversion.
            The more the faster conversion.
            Don't use too much or it will crash.
    """

    assert len(images) == len(labels)

    data_list = []

    start_time = time.time()

    # apply better istarmap trick (enables progress bar when using multiprocessing)
    multiprocessing.pool.Pool.istarmap = multiprocessing_istarmap

    with multiprocessing.Pool(num_workers) as pool:
        args = list(zip(images, labels, repeat(desired_nodes), repeat(slic_kwargs)))

        for graph in tqdm(
            pool.istarmap(convert_numpy_img_to_superpixel_graph, args),
            total=len(args),
            desc="Generating superpixels",
            colour="GREEN",
        ):
            x, edge_index, pos, y = graph
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, pos=pos, y=y))

    total_time = time.time() - start_time
    print(f"Took {total_time}s.")

    return data_list


def convert_numpy_img_to_superpixel_graph(img, label, desired_nodes: int = 75, slic_kwargs={}):
    """Convert numpy img to superpixel grap.
    Each superpixel is connected to all superpixels it directly touches.

    Args:
        img (np array): Numpy image array.
        desired_nodes (int): Desired number of superpixels.
            The actual number is usually a bit different due to the
            way SLIC algorithm behaves.

    Returns:
        Tuple: Tuple of node features, edge index and node positions
    """

    img = img / 255

    height = img.shape[0]
    width = img.shape[1]
    num_of_features = img.shape[2]

    segments = slic(img, n_segments=desired_nodes, slic_zero=True, start_label=0, **slic_kwargs)

    num_of_nodes = np.max(segments) + 1
    nodes = {node: {"rgb_list": [], "pos_list": []} for node in range(num_of_nodes)}

    # get rgb values and positions
    for y in range(height):
        for x in range(width):
            node = segments[y, x]

            rgb = img[y, x, :]
            nodes[node]["rgb_list"].append(rgb)

            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["pos_list"].append(pos)

    # compute features
    G = nx.Graph()
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        G.add_node(node, features=list(rgb_mean))

    # compute node positions
    segments_ids = np.unique(segments)
    pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    pos = pos.astype(int)

    # add edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])

    # add self loops
    for node in nodes:
        G.add_edge(node, node)

    # get edge_index
    m = len(G.edges)
    edge_index = np.zeros([2 * m, 2]).astype(np.int64)
    for e, (s, t) in enumerate(G.edges):
        edge_index[e, 0] = s
        edge_index[e, 1] = t
        edge_index[m + e, 0] = t
        edge_index[m + e, 1] = s

    # get features
    x = np.zeros([num_of_nodes, num_of_features]).astype(np.float32)
    for node in G.nodes:
        x[node, :] = G.nodes[node]["features"]

    y = label

    return x, edge_index, pos, y
