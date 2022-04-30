import torch.nn as nn


class FactorizedReduce(nn.Module):


    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert(C_out % 2 == 0)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


if __name__ == "__main__":
    import torch
    net = FactorizedReduce(3, 6)

    x = torch.rand(4, 3, 64, 64)

    print(net(x).shape)

