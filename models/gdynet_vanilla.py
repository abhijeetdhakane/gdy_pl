from typing import Union, Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size

import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


class CGConv(MessagePassing):
    """
    Crystal Graph Convolution layer with edge-conditioned filtering.

    This layer follows the CGCNN-style message passing paradigm, where
    messages are constructed from:
        - central atom features (x_i)
        - neighboring atom features (x_j)
        - bond / neighbor features (edge_attr)

    An attention-like scalar filter is learned per edge and applied
    during aggregation.
    """

    def __init__(self,
                 nbr_fea_len,
                 atom_fea_len=64,
                 aggr: str = 'mean',
                 bias: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        nbr_fea_len : int
            Dimension of neighbor / bond feature vector.
        atom_fea_len : int, default=64
            Dimension of atom feature embeddings.
        aggr : str, default='mean'
            Aggregation method for message passing.
        bias : bool, default=True
            Whether to include bias terms in linear layers.
        """
        super(CGConv, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        # BatchNorm applied to message transformation and output
        self.bn1 = nn.BatchNorm1d(self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

        # Linear layer producing the transformed message content
        self.lin_core = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            self.atom_fea_len,
            bias=bias
        )

        # Linear layer producing a scalar attention/filter weight per edge
        self.lin_filter = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            1,
            bias=bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes learnable parameters.

        Xavier initialization is used for linear layers,
        while batch normalization parameters are reset to defaults.
        """
        torch.nn.init.xavier_uniform_(self.lin_core.weight)
        torch.nn.init.xavier_uniform_(self.lin_filter.weight)

        self.lin_core.bias.data.fill_(0)
        self.lin_filter.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the convolution layer.

        Parameters
        ----------
        x : torch.Tensor
            Atom feature matrix with shape [num_atoms, atom_fea_len].
        edge_index : torch.Tensor
            Edge connectivity in COO format.
        edge_attr : torch.Tensor
            Edge (bond) feature matrix.

        Returns
        -------
        torch.Tensor
            Updated atom feature matrix.
        """
        # Perform message passing and aggregation
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr,
            size=(x.size(0), x.size(0))
        )

        # Residual connection with original atom features
        out = nn.ReLU()(self.bn2(out) + x)

        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages for each edge.

        Parameters
        ----------
        x_i : torch.Tensor
            Feature vector of the target atom.
        x_j : torch.Tensor
            Feature vector of the source atom.
        edge_attr : torch.Tensor
            Bond / neighbor features.

        Returns
        -------
        torch.Tensor
            Concatenated message tensor.
        """
        # Concatenate target atom, source atom, and bond features
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        return z

    def aggregate(self, inputs, index):
        """
        Aggregates messages using a learned attention/filter mechanism.

        Parameters
        ----------
        inputs : torch.Tensor
            Edge messages produced by `message()`.
        index : torch.Tensor
            Target node indices for each message.

        Returns
        -------
        torch.Tensor
            Aggregated node features.
        """
        # Scalar filter for attention weighting
        lin_filt = self.lin_filter(inputs)

        # Core message transformation
        lin_core = nn.ReLU()(self.bn1(self.lin_core(inputs)))

        # Normalize filter weights across neighbors of each target node
        alpha = softmax(lin_filt, index)

        # Weighted aggregation of messages
        out_agg = scatter(alpha * lin_core, index, dim=0, reduce='mean')

        return out_agg


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN-like architecture).

    Consists of:
        - Atom embedding layer
        - Multiple CGConv message passing layers
        - Target-based pooling
        - Fully-connected classification head

    Conceptual and methodological inspiration:
    Xie, T., France-Lanord, A., Wang, Y., Shao-Horn, Y., & Grossman, J. C. (2019).
    "Graph dynamical networks for unsupervised learning of atomic scale dynamics in materials."
    Nature Communications, 10, 2667.
    """


    def __init__(self,
                 nbr_fea_len,
                 atom_fea_len=64,
                 n_conv=3,
                 state_len=2):
        """
        Parameters
        ----------
        nbr_fea_len : int
            Dimension of neighbor / bond features.
        atom_fea_len : int, default=64
            Dimension of atom embeddings.
        n_conv : int, default=3
            Number of graph convolution layers.
        state_len : int, default=2
            Dimension of output state vector.
        """
        super(CrystalGraphConvNet, self).__init__()

        # Atom-type embedding: integer atom type -> dense vector
        self.embedding = nn.Embedding(100, atom_fea_len)

        # Stack of graph convolution layers
        self.convs = nn.ModuleList([
            CGConv(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len
            )
            for _ in range(n_conv)
        ])

        # Fully-connected layers for final prediction
        self.conv_to_fc = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc_state = nn.Linear(atom_fea_len, state_len)

    def forward(self, data):
        """
        Forward pass of the CGCNN model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data object with attributes:
                - x : atom types
                - edge_index : bond connectivity
                - edge_attr : bond features
                - target : indices of atoms used for pooling

        Returns
        -------
        torch.Tensor
            Softmax-normalized state probabilities.
        """
        atom_fea, bond_index, bond_attr, target_index = (
            data.x, data.edge_index, data.edge_attr, data.target
        )

        # Embed atom types into feature vectors
        atom_fea = self.embedding(atom_fea)

        # Apply graph convolution layers
        for conv_func in self.convs:
            atom_fea = conv_func(
                x=atom_fea,
                edge_index=bond_index,
                edge_attr=bond_attr
            )

        # Pool features from selected target atoms
        crys_fea = torch.index_select(atom_fea, 0, target_index)

        # Fully-connected prediction head
        crys_fea = nn.ReLU()(crys_fea)
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = nn.ReLU()(crys_fea)
        crys_fea = self.fc_state(crys_fea)

        # Output class probabilities
        out = nn.Softmax(dim=-1)(crys_fea)

        return out
