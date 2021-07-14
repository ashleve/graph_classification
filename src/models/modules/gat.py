from torch import nn
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool


class GAT(nn.Module):
    """Flexible GAT network."""

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        for param_name in [
            "num_node_features",
            "num_conv_layers",
            "conv_size",
            "lin1_size",
            "lin2_size",
            "output_size",
        ]:
            if not isinstance(hparams[param_name], int):
                raise Exception("Wrong hyperparameter type!")

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        if hparams["activation"] == "relu":
            activation = nn.ReLU
        elif hparams["activation"] == "prelu":
            activation = nn.PReLU
        else:
            raise Exception("Invalid activation function name.")

        if hparams["pool_method"] == "add":
            self.pooling_method = global_add_pool
        elif hparams["pool_method"] == "mean":
            self.pooling_method = global_mean_pool
        elif hparams["pool_method"] == "max":
            self.pooling_method = global_max_pool
        else:
            raise Exception("Invalid pooling method name")

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        heads = hparams.get("heads", 1)

        self.conv_modules.append(
            GATConv(hparams["num_node_features"], hparams["conv_size"], heads=heads)
        )
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = GATConv(heads * hparams["conv_size"], hparams["conv_size"], heads=heads)
            # nn.init.xavier_uniform_(conv.lin_l.weight)
            # nn.init.xavier_uniform_(conv.lin_r.weight)
            # nn.init.xavier_uniform_(conv.att_l)
            # nn.init.xavier_uniform_(conv.att_r)
            self.conv_modules.append(conv)
            self.activ_modules.append(activation())

        self.lin1 = nn.Linear(heads * hparams["conv_size"], hparams["lin1_size"])
        self.activ_lin1 = activation()

        self.lin2 = nn.Linear(hparams["lin1_size"], hparams["lin2_size"])
        self.activ_lin2 = activation()

        self.output = nn.Linear(hparams["lin2_size"], hparams["output_size"])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)

        x = self.pooling_method(x, batch)

        x = self.lin1(x)
        x = self.activ_lin1(x)

        x = self.lin2(x)
        x = self.activ_lin2(x)

        return self.output(x)
