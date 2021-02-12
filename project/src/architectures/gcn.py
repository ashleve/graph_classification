from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn


class GCN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.conv1 = GCNConv(hparams['node_features'], hparams['conv1_size'])
        self.conv2 = GCNConv(hparams['conv1_size'], hparams['conv2_size'])
        self.conv3 = GCNConv(hparams['conv2_size'], hparams['conv3_size'])

        self.linear1 = nn.Linear(hparams['conv3_size'], hparams['lin1_size'])
        # self.dropout1 = nn.Dropout(p=hparams['dropout1'])
        self.linear2 = nn.Linear(hparams['lin1_size'], hparams['lin2_size'])
        # self.dropout2 = nn.Dropout(p=hparams['dropout2'])
        self.linear3 = nn.Linear(hparams['lin2_size'], hparams['output_size'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        if self.hparams['pool_method'] == 'add':
            x = global_add_pool(x, data.batch)
        elif self.hparams['pool_method'] == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.hparams['pool_method'] == 'max':
            x = global_max_pool(x, data.batch)
        else:
            raise Exception("Invalid pooling method")

        x = self.linear1(x)
        x = F.relu(x)
        # self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        # self.dropout2(x)
        x = self.linear3(x)

        return x
