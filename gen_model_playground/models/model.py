import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

class Model(nn.Module):
    """
    A custom neural network model.

    Attributes:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_blocks: Number of blocks in the model.
        hidden_features: Dimension of hidden layers.
        cond_features: Additional condition feature dimension.
        spectral: Flag to use spectral normalization.
        batch_norm: Flag to use batch normalization.
        residual: Flag to use residual connections.
        time_features: Dimension of time-related features.
        bias: Flag to use bias in linear layers.
    """
    def __init__(self, in_features: int, out_features: int, num_blocks: int = 4, hidden_features: int = 64, cond_features=0, spectral=False, batch_norm=True, residual=True, time_features=0, bias=False, dropout=0.0,**kwargs):
        super(Model, self).__init__()

        # Time feature flag
        self.time = time_features > 0

        # Initializing input block with optional spectral normalization
        self.inblock = nn.Linear(in_features + int(cond_features) + int(time_features), hidden_features, bias=bias)
        if spectral:
            self.inblock = spectral_norm(self.inblock)

        # Creating a list of middle blocks
        self.midblocks = nn.ModuleList([Block(hidden_features, spectral, batch_norm, bias, time_features,int(cond_features),dropout) for _ in range(num_blocks)])

        # Initializing output block with optional spectral normalization
        self.outblock = nn.Linear(hidden_features, out_features, bias=bias)
        if spectral:
            self.outblock = spectral_norm(self.outblock)

        # Activation function
        self.act = lambda x: x * torch.nn.functional.sigmoid(x)

        # Residual connection flag
        self.residual = residual

    def forward(self, x: torch.Tensor, t=None,  cond=None, feature_matching=False) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor.
            t: Time-related feature tensor (optional).
            cond: Additional condition tensor (optional).
            feature_matching: Flag to return intermediate values for feature matching.

        Returns:
            Output tensor, and optionally intermediate tensor if feature_matching is True.
        """
        # Processing time features
        if self.time and t is not None:
            if len(t)==1:
                t = t.repeat(x.shape[0], 1)
            if len(t.shape)==1:
                t = t.unsqueeze(1)
            x = torch.cat((x, t), dim=-1)

        # Processing conditional features
        if cond is not None:
            if len(cond.shape)==1:
                cond = cond.unsqueeze(1)
            x = torch.cat((x, cond), dim=-1)

        # Input block
        val = self.inblock(x)

        # Iterating through middle blocks
        for midblock in self.midblocks:

            val = val + midblock(val, t=t, cond=cond) if self.residual else midblock(val, t=t, cond=cond)

        # Activation and output block
        x = self.act(val)
        x = self.outblock(x)

        # Feature matching condition
        return (x, val) if feature_matching else x


class Block(nn.Module):
    """
    A block in the neural network model.

    Attributes:
        hidden_features: Dimension of hidden layers.
        spectral: Flag to use spectral normalization.
        batch_norm: Flag to use batch normalization.
        bias: Flag to use bias in linear layers.
        time_features: Dimension of time-related features.
    """
    def __init__(self, hidden_features, spectral, batch_norm, bias, time_features=0, cond_features=0,dropout=0.0):
        super(Block, self).__init__()

        # Initializing first linear layer with optional spectral normalization
        self.linear = nn.Linear(hidden_features + time_features + cond_features, hidden_features, bias=bias)
        if spectral:
            self.linear = spectral_norm(self.linear)

        # Initializing second linear layer with optional spectral normalization
        self.linear2 = nn.Linear(hidden_features, hidden_features, bias=bias)
        if spectral:
            self.linear2 = spectral_norm(self.linear2)

        # Activation function
        self.act = lambda x: x * torch.nn.functional.sigmoid(x)
        self.dropout = nn.Dropout(dropout)
        # Batch normalization or identity layers
        self.bn = nn.BatchNorm1d(hidden_features + cond_features, track_running_stats=True) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden_features, track_running_stats=True) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor, t=None, cond=None):
        """
        Forward pass of the block.

        Args:
            x: Input tensor.
            t: Time-related feature tensor (optional).

        Returns:
            Processed tensor.
        """
        # Adding cond
        if cond is not None:
            if len(cond.shape)==1:
                cond = cond.unsqueeze(1)
            x = torch.cat((x, cond), dim=-1)
        # Processing input with batch normalization and activation
        x = self.linear(self.act(self.bn(x))) if t is None else self.linear(self.act(torch.cat((self.bn(x), t), dim=-1)))
        x = self.dropout(x)
        # Second linear layer with batch normalization and activation
        x = self.linear2(self.act(self.bn2(x)))

        return x