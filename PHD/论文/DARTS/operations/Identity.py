import torch.nn as nn


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# test
if __name__ == '__main__':
    import torch

    x = torch.rand(2, 3, 43, 43)

    net = Identity()

    assert(torch.allclose(net(x), x))
