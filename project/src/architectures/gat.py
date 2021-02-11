from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GAT(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.atom_encoder = AtomEncoder(emb_dim=300)

        self.conv1 = GATConv(300, hparams['conv1_size'], heads=1)
        self.conv2 = GATConv(hparams['conv1_size'], hparams['conv2_size'], heads=1)
        self.conv3 = GATConv(hparams['conv2_size'], hparams['conv3_size'], heads=1)

        self.linear1 = nn.Linear(hparams['conv3_size'], hparams['lin1_size'])
        self.dropout1 = nn.Dropout(p=hparams['dropout1'])
        self.linear2 = nn.Linear(hparams['lin1_size'], hparams['output_size'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.atom_encoder(x)  # x is input atom feature

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

        return x
