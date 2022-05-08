import torch
import torch.nn as nn




class Encoder(nn.Module):


    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(2, ))
        self.w = nn.Linear(1, 1, bias=False)

        self.relax = nn.Softmax(0)

        

    
    def forward(self, x):
        a1, a2 = self.relax(self.alpha)

        x = (2*a1*x + 4*a2*x).mean(dim=(0, 1, 2, 3, 4)).unsqueeze(0)
        
        return self.w(x)


    def update_w(self):
        self.w.requires_grad_(True)
        self.alpha.requires_grad_(False)


    def update_alpha(self):
        self.w.requires_grad_(False)
        self.alpha.requires_grad_(True)

class Test(nn.Module):


    def __init__(self):
        super().__init__()
        self.cell = Encoder()
        self.block = Encoder()

    def forward(self, x):
        return self.cell(x) + self.block(x)
    

    def update_w(self):
        self.cell.update_w()
        self.block.update_w()


    def update_alpha(self):
        self.cell.update_alpha()
        self.block.update_alpha()


def get_params_to_update(model):
    """ Returns list of model parameters that have required_grad=True"""
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update

model = Test()

# set the weight optimizer
model.update_w()
w_opt = torch.optim.SGD(get_params_to_update(model), lr=1)

# set the alpha optimizer
model.update_alpha()
a_opt = torch.optim.SGD(get_params_to_update(model), lr=1)



x = torch.rand(2, 1, 64, 64, 64)

model.update_w()
o = model(x)
o.backward()

print(model.cell.alpha)
w_opt.step()
print(model.cell.alpha)


model.update_alpha()
o = model(x)
o.backward()

print(model.cell.alpha)
a_opt.step()
print(model.cell.alpha)
