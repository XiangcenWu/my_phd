from torch import conv2d
import torch.nn as nn


class SepConv(nn.Module):


    def __init__(
        self,
        C_in :int,
        C_out :int,
        kernel_size :int or tuple,
        stride :int,
        padding :int,
        affine :bool=True
    ):
        """Sep

        Args:
            C_in (int): _description_
            C_out (int): _description_
            kernel_size (intortuple): _description_
            stride (int): _description_
            padding (int): _description_
            affine (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.op = nn.Sequential(
            # RELU -> CONV -> CONV -> BN
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            # RELU -> CONV -> CONV -> BN
            nn.ReLU(),
            # ################################
            # The stride of this operation must be one in order to make sure the size of the feature map is halfed 1/2 not 1/4
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            ##################################
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    

    def forward(self, x):
        return self.op(x)
    

if __name__ == "__main__":
    import torch

    input_channel = 3
    output_channel = 6
    kernel_size = 3 # 5
    stride = 1
    padding = 1

    net = SepConv(
        input_channel,
        output_channel,
        kernel_size,
        stride,
        padding
    )


    x = torch.rand(4, 3, 64, 64)

    print(net(x).shape)


