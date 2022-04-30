import torch.nn as nn


class DilConv(nn.Module):


    def __init__(
        self,
        C_in :int,
        C_out :int,
        kernel_size :int or tuple,
        stride :int,
        padding :int,
        dilation :int,
        affine :bool=True
    ):
        """Dilated convolution block (ReLU ->  -> BatchNorm)

        Args:
            C_in (int): input channel size
            C_out (int): output channel size
            kernal_size (int): kernal size of the convolution operation
            stride (int): stride of the cell at high level
            padding (int): padding of the convolution operation
            dilation (int): dilation of the convolution operation
            affine (bool, optional): affine of the BatchNorm operation. Defaults to True.
        """

        super().__init__()

        # The actual operations
        self.op = nn.Sequential(
            # Relu
            nn.ReLU(),
            # conv2d
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),

            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


if __name__ == "__main__":
    import torch
    
    net = DilConv(2, 3, (3, 3), 2, 2, 2)

    x = torch.rand(4, 2, 36, 36)

    o = net(x)

    print(o.shape)
