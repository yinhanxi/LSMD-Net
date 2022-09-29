import torch

class SparseToDensePool(torch.nn.Module):
    '''
    Converts sparse inputs to dense outputs using max and min pooling
    with different kernel sizes and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max and/or average pool(s)
        min_pool_sizes : list[int]
            list of min pool sizes s (kernel size is s x s)
        max_pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_filter : int
            number of filters for 1 x 1 convolutions
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 min_pool_sizes=[3, 5, 7, 9],
                 max_pool_sizes=[3, 5, 7, 9],
                 n_filter=8,
                 n_convolution=3,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(SparseToDensePool, self).__init__()

        activation_func = activation_func(activation_func)

        self.min_pool_sizes = [
            s for s in min_pool_sizes if s > 1
        ]

        self.max_pool_sizes = [
            s for s in max_pool_sizes if s > 1
        ]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        pool_convs = []
        for n in range(n_convolution):
            conv = Conv2d(
                in_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            pool_convs.append(conv)

            # Set new input channels as output channels
            in_channels = n_filter

        self.pool_convs = torch.nn.Sequential(*pool_convs)

        in_channels = n_filter + input_channels

        self.conv = Conv2d(
            in_channels,
            n_filter,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False)

    def forward(self, x):
        # Input depth
        z = torch.unsqueeze(x[:, 0, ...], dim=1)

        pool_pyramid = []

        # Use min and max pooling to densify and increase receptive field
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            # Set flag (999) for any zeros and max pool on -z then revert the values
            z_pool = -pool(torch.where(z == 0, -999 * torch.ones_like(z), -z))
            # Remove any 999 from the results
            z_pool = torch.where(z_pool == 999, torch.zeros_like(z), z_pool)

            pool_pyramid.append(z_pool)

        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(z)

            pool_pyramid.append(z_pool)

        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        pool_convs = self.pool_convs(pool_pyramid)

        pool_convs = torch.cat([pool_convs, x], dim=1)

        return self.conv(pool_convs)


class Conv2d(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(Conv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv = self.conv(x)
        conv = self.batch_norm(conv) if self.use_batch_norm else conv

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv

def activation_func(activation_fn):

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))
