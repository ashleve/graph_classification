from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
from torch import nn
import torch


class GCN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.conv1 = GCNConv(hparams['num_node_features'], hparams['conv1_size'])
        self.conv2 = GCNConv(hparams['conv1_size'], hparams['conv2_size'])
        self.conv3 = GCNConv(hparams['conv2_size'], hparams['conv3_size'])

        self.linear1 = nn.Linear(hparams['conv3_size'], hparams['lin1_size'])
        self.dropout1 = nn.Dropout(p=hparams['dropout1'])
        self.linear2 = nn.Linear(hparams['lin1_size'], hparams['num_classes'])

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

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        return F.log_softmax(x, dim=0)


class DGCNN(nn.Module):
    def __init__(self, hparams):
        """https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf"""
        super().__init__()
        self.hparams = hparams
        self.conv1 = GCNConv(hparams['num_node_features'], 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(1, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)

        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(352, 128)
        self.drop_out = Dropout(0.5)
        self.classifier_2 = Linear(128, hparams['num_classes'])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)

        x_1 = F.tanh(self.conv1(x, edge_index))
        x_2 = F.tanh(self.conv2(x_1, edge_index))
        x_3 = F.tanh(self.conv3(x_2, edge_index))
        x_4 = F.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch, k=30)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)

        return classes
