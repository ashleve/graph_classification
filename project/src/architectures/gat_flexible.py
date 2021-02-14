from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import GATConv
from torch import nn


class GAT(nn.Module):
    """Flexible GAT network for hyperparameter search."""

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

        self.conv1 = GATConv(hparams['node_features'], hparams['conv1_size'], heads=1)
        self.activ_conv1 = self.activation()

        self.conv2 = GATConv(hparams['conv1_size'], hparams['conv2_size'], heads=1)
        self.activ_conv2 = self.activation()

        self.conv3 = None
        if type(hparams.get('conv3_size')) is int:
            self.conv3 = GATConv(hparams['conv2_size'], hparams['conv3_size'], heads=1)
            self.activ_conv3 = self.activation()

        self.conv4 = None
        if type(hparams.get('conv3_size')) is int and type(hparams.get('conv4_size')) is int:
            self.conv4 = GATConv(hparams['conv3_size'], hparams['conv4_size'], heads=1)
            self.activ_conv4 = self.activation()

        if self.hparams['pool_method'] == 'add':
            self.pooling_method = global_add_pool
        elif self.hparams['pool_method'] == 'mean':
            self.pooling_method = global_mean_pool
        elif self.hparams['pool_method'] == 'max':
            self.pooling_method = global_max_pool
        else:
            raise Exception("Invalid pooling method name")

        if self.conv3 and self.conv4:
            lin1_input_size = hparams['conv4_size']
        elif self.conv3 and not self.conv4:
            lin1_input_size = hparams['conv3_size']
        elif not self.conv3 and not self.conv4 and self.conv2:
            lin1_input_size = hparams['conv2_size']
        else:
            raise Exception("wtf")

        self.batch_norm = None
        if self.hparams.get("use_batch_norm_after_pooling"):
            self.batch_norm = nn.BatchNorm1d(lin1_input_size)

        self.linear1 = nn.Linear(lin1_input_size, hparams['lin1_size'])
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

        if self.conv3:
            x = self.conv3(x, edge_index)
            x = self.activ_conv3(x)

        if self.conv4:
            x = self.conv4(x, edge_index)
            x = self.activ_conv4(x)

        x = self.pooling_method(x, data.batch)

        if self.batch_norm:
            x = self.batch_norm(x)

        x = self.linear1(x)
        x = self.activ_lin1(x)

        x = self.linear2(x)
        x = self.activ_lin2(x)

        return self.output(x)
