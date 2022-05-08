import torch
import torch.nn as nn
# xiehaolehfdjsaklfhjdsaklfhasd

def get_params_to_update(model):
    """ Returns list of model parameters that have required_grad=True"""
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


class Encoder(nn.Module):
    
    
    def __init__(self, in_c, out_c):
        super().__init__()

        # Block One
        self.block1 = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(in_c, out_c, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_c)
        )
        # Block Two
        self.block2 = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.Conv3d(in_c, out_c, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_c)
        )
        # Block three
        self.block3 = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(in_c, out_c, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(out_c)
        )
        # softmax for alpha
        self.relax = nn.Softmax(dim=0)
        # alpha parameters
        self.alpha = nn.Parameter(torch.zeros((3, )))



    def forward(self, x):
        a1, a2, a3 = self.relax(self.alpha)
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        return a1*x1 + a2*x2 + a3*x3


    def update_alpha(self):
        # set the model parameters requires grad=False
        # self.block1.requires_grad_(False)
        # self.block2.requires_grad_(False)
        # self.block3.requires_grad_(False)
        for paras in self.parameters():
            paras.requires_grad_(False)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(True)


    def update_w(self):
        # set the model parameters requires grad=False
        # self.block1.requires_grad_(True)
        # self.block2.requires_grad_(True)
        # self.block3.requires_grad_(True)
        for paras in self.parameters():
            paras.requires_grad_(True)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(False)





class Decoder(nn.Module):
    
    
    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass


    def update_alpha(self):
        pass


    def update_w(self):
        pass



class Network(nn.Module):
    
    
    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass


    def update_alpha(self):
        pass


    def update_w(self):
        pass


model = Encoder(1, 3)
x = torch.rand(2, 1, 64, 64, 64)



# set the weight optimizer
model.update_w()
w_opt = torch.optim.SGD(get_params_to_update(model), lr=1)

# set the alpha optimizer
model.update_alpha()
a_opt = torch.optim.SGD(get_params_to_update(model), lr=1)



model.update_alpha()
o = model(x).sum()
o.backward()

print(model.alpha)
print(model.block3[0].weight.data)
a_opt.step()
print(model.block3[0].weight.data)
print(model.alpha)


print('******************************')
model.update_w()
print(model.alpha)
print(model.block3[0].weight.data)

o = model(x).sum()
o.backward()
w_opt.step()
print(model.alpha)
print(model.block3[0].weight.data)