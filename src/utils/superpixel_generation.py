import multiprocessing
import os
import time
from itertools import repeat

import networkx as nx
import numpy as np
import torch
from better_istarmap import better_istarmap
from skimage.segmentation import slic
from torch_geometric.data import Data

# tqdm for progress bar
from tqdm.auto import tqdm

# apply patch to enable progress bar with multiprocessing,
# requires python 3.8+
# see https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
# from src.utils.better_istarmap import better_istarmap


def save_torch_geometric_superpixel_dataset(data_dir, dataset_name: str, data: dict, slices: dict):
    dataset = Data(x=data["x"], edge_index=data["edge_index"], pos=data["pos"], y=data["y"])

    path = os.path.join(data_dir, dataset_name, "processed", "data.pt")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path)

    torch.save((dataset, slices), path)


def convert_torchvision_dataset_to_superpixel_graphs(
    dataset, desired_nodes: int = 75, num_workers: int = 4
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

    images, labels = dataset

    # first we gather all graphs into lists, then we convert them into torch matrixes
    x_list = []
    edge_index_list = []
    pos_list = []

    # slices to know when given graphs starts and stops in a dataset matrix
    x_slices_list = [0]
    edge_index_slices_list = [0]
    pos_slices_list = [0]
    y_slices_list = [0]

    print("Processing images into graphs...")
    start_time = time.time()

    # apply better istarmap trick (enables progress bar when using multiprocessing)
    multiprocessing.pool.Pool.istarmap = better_istarmap

    with multiprocessing.Pool(num_workers) as pool:
        args = list(zip(images, repeat(desired_nodes)))
        for graph in tqdm(
            pool.istarmap(convert_numpy_img_to_superpixel_graph, args), total=len(args)
        ):
            x, edge_index, pos = graph

            x_list.append(torch.as_tensor(x, dtype=torch.float32))
            edge_index_list.append(torch.as_tensor(edge_index, dtype=torch.long))
            pos_list.append(torch.as_tensor(pos, dtype=torch.float32))

            x_slices_list.append(x_slices_list[-1] + len(x))
            edge_index_slices_list.append(edge_index_slices_list[-1] + len(edge_index))
            pos_slices_list.append(pos_slices_list[-1] + len(pos))
            y_slices_list.append(y_slices_list[-1] + 1)

    total_time = time.time() - start_time
    print(f"Took {total_time}s.")

    data = {}
    data["x"] = torch.cat(x_list, dim=0)
    data["edge_index"] = torch.cat(edge_index_list, dim=0).T
    data["pos"] = torch.cat(pos_list, dim=0)
    data["y"] = torch.as_tensor(labels, dtype=torch.long)

    del x_list, edge_index_list, pos_list

    slices = {}
    slices["x"] = torch.as_tensor(x_slices_list, dtype=torch.long)
    slices["edge_index"] = torch.as_tensor(edge_index_slices_list, dtype=torch.long)
    slices["pos"] = torch.as_tensor(pos_slices_list, dtype=torch.long)
    slices["y"] = torch.as_tensor(y_slices_list, dtype=torch.long)

    del x_slices_list, edge_index_slices_list, pos_slices_list, y_slices_list

    return data, slices


def convert_numpy_img_to_superpixel_graph(img, desired_nodes: int = 75):
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

    segments = slic(img, n_segments=desired_nodes, slic_zero=True, start_label=0)

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

    return x, edge_index, pos


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

    trainset = FashionMNIST("data/", download=True, train=True)
    testset = FashionMNIST("data/", download=True, train=False)

    train_images = trainset.data
    test_images = testset.data
    print(train_images.shape)
    print(test_images.shape)

    images = np.concatenate((train_images, test_images))

    # don't do reshape for cifar10!
    images = np.reshape(images, (len(images), 28, 28, 1))
    print(images.shape)

    train_labels = trainset.targets
    test_labels = testset.targets
    labels = np.concatenate((train_labels, test_labels))
    print(labels.shape)

    dataset = (images, labels)
    data, slices = convert_torchvision_dataset_to_superpixel_graphs(
        dataset, desired_nodes=100, num_workers=10
    )

    torch_dataset = Data(x=data["x"], edge_index=data["edge_index"], pos=data["pos"], y=data["y"])
    print(torch_dataset)
