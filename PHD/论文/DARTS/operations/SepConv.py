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
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    

    def forward(self, x):
        return self.op(x)
        
