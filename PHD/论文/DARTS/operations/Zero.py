
import torch.nn as nn


class Zero(nn.Module):
    def __init__(self, stride :int):
        """Zero operation

        Args:
            stride (int): Stride of the Cell
        """
        super().__init__()
        self.stride = stride

    def forward(self, x):

        # if the cell is of stride one, 
        # then the resolution of feature map
        # is not changed 
        if self.stride == 1:
            return x.mul(0.)
        # if stride 2, then return half 
        # of the original resolution
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


# Test this operation
if __name__ == '__main__':
    import torch

    x = torch.rand(2, 1, 32, 32)

    net = Zero(1)

    o = net(x)

    assert(torch.allclose(torch.zeros_like(x), o))

    net_stride2 = Zero(2)

    o2 = net_stride2(x)

    assert(torch.allclose(torch.zeros((2, 1, 16, 16)), o2))
