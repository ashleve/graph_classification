from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv
from torch import nn


class GCN(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if not (isinstance(hparams['node_features'], int)
                and isinstance(hparams['conv1_size'], int)
                and isinstance(hparams['conv2_size'], int)
                and isinstance(hparams['lin1_size'], int)
                and isinstance(hparams['lin2_size'], int)
                and isinstance(hparams['output_size'], int)):
            raise Exception("Wrong type!")

        if self.hparams['activation'] == "relu":
            self.activation = nn.ReLU
        elif self.hparams['activation'] == "prelu":
            self.activation = nn.PReLU
        else:
            raise Exception("Invalid activation function name")

        self.conv1 = GCNConv(hparams['node_features'], hparams['conv1_size'])
        self.activ_conv1 = self.activation()
        self.conv2 = GCNConv(hparams['conv1_size'], hparams['conv2_size'])
        self.activ_conv2 = self.activation()
        self.conv3 = GCNConv(hparams['conv2_size'], hparams['conv3_size'])
        self.activ_conv3 = self.activation()

        if self.hparams['pool_method'] == 'add':
            self.pooling_method = global_add_pool
        elif self.hparams['pool_method'] == 'mean':
            self.pooling_method = global_mean_pool
        elif self.hparams['pool_method'] == 'max':
            self.pooling_method = global_max_pool
        else:
            raise Exception("Invalid pooling method name")

        self.linear1 = nn.Linear(hparams['conv3_size'], hparams['lin1_size'])
        self.activ_lin1 = self.activation()
        self.linear2 = nn.Linear(hparams['lin1_size'], hparams['lin2_size'])
        self.activ_lin2 = self.activation()

        self.output = nn.Linear(hparams['lin2_size'], hparams['output_size'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.activ_conv1(x)
        x = self.conv2(x, edge_index)
        x = self.activ_conv2(x)
        x = self.conv3(x, edge_index)
        x = self.activ_conv3(x)

        x = self.pooling_method(x, data.batch)

        x = self.linear1(x)
        x = self.activ_lin1(x)
        x = self.linear2(x)
        x = self.activ_lin2(x)

        return self.output(x)
