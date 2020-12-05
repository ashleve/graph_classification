from skimage.segmentation import slic
import networkx as nx
import numpy as np

# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp


def convert_img_to_superpixels_graph(image, desired_nodes=75, add_position_to_features=True):
    height = image.shape[0]
    width = image.shape[1]
    num_of_features = image.shape[2] + 2 if add_position_to_features else image.shape[2]

    segments = slic(image, n_segments=desired_nodes, slic_zero=True, start_label=0)

    num_of_nodes = np.max(segments) + 1
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_of_nodes)
    }

    # get rgb
    for y in range(height):
        for x in range(width):
            node = segments[y, x]

            rgb = image[y, x, :]
            nodes[node]["rgb_list"].append(rgb)

            pos = np.array([float(x) / width, float(y) / height])
            nodes[node]["pos_list"].append(pos)

    # compute features (from rgb only)
    G = nx.Graph()
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        if add_position_to_features:
            features = np.concatenate((rgb_mean, pos_mean))
        else:
            features = rgb_mean
        G.add_node(node, features=list(features))

    # compute node positions
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    centers = centers.astype(int)

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

    return x, edge_index, centers


def better_istarmap(self, func, iterable, chunksize=1):
    """
        starmap-version of imap
        This is only for possibility of displaying progress bar in jupyter notebook during
        multiprocessing of images to superpixels graphs.
        Source: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(
            result._job,
            mpp.starmapstar,
            task_batches
        ),
        result._set_length
    ))
    return (item for chunk in result for item in chunk)

# to apply patch:
# mpp.Pool.istarmap = better_istarmap
