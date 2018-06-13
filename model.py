import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision

class globalnet(nn.Module):
    def __init__(self, opt):
        super(globalnet, self).__init__()
        self.opt = opt

        if self.opt.model == 'resnet50':
            self.resnet50 = torchvision.models.resnet50(pretrained=True)
            self.removed = list(self.resnet50.children())[:-2]   
            self.resnet_layer = nn.Sequential(*self.removed)
            self.pool_layer = nn.MaxPool2d(7)  
            self.Linear_layer = nn.Linear(2048, 14)
            self.sigmoid = nn.Sigmoid()

        if self.opt.model == 'densenet121':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
            self.removed = list(self.densenet121.children())[:-1]   
            self.resnet_layer = nn.Sequential(*self.removed)
            self.pool_layer = nn.MaxPool2d(7)  
            self.Linear_layer = nn.Linear(1024, 14)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        heatmap = self.resnet_layer(x)
        pool = self.pool_layer(heatmap)
        x = pool.view(pool.size(0), -1) 
        x = self.sigmoid(self.Linear_layer(x))        
        return x, heatmap.data, pool.data

class localnet(nn.Module):
    def __init__(self, opt):
        super(localnet, self).__init__()
        self.opt = opt

        if self.opt.model == 'resnet50':
            self.resnet50 = torchvision.models.resnet50(pretrained=True)
            self.removed = list(self.resnet50.children())[:-2]   
            self.resnet_layer = nn.Sequential(*self.removed)
            self.pool_layer = nn.MaxPool2d(7)  
            self.Linear_layer = nn.Linear(2048, 14)
            self.sigmoid = nn.Sigmoid()

        if self.opt.model == 'densenet121':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
            self.removed = list(self.densenet121.children())[:-1]   
            self.resnet_layer = nn.Sequential(*self.removed)
            self.pool_layer = nn.MaxPool2d(7)  
            self.Linear_layer = nn.Linear(1024, 14)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        heatmap = self.resnet_layer(x)
        pool = self.pool_layer(heatmap)
        x = pool.view(pool.size(0), -1) 
        x = self.sigmoid(self.Linear_layer(x))        
        return x, heatmap.data, pool.data

class fusionnet(nn.Module):
    def __init__(self, opt):
        super(fusionnet, self).__init__()
        self.opt = opt 

        if self.opt.model == 'resnet50':
            self.Linear_layer = nn.Linear(2048*2, 14)

        if self.opt.model == 'densenet121':
            self.Linear_layer = nn.Linear(1024*2, 14)
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.sigmoid(self.Linear_layer(x))        
        return x, x, x

if __name__ == '__main__':
	resnet50 = torchvision.models.resnet50(pretrained = False)
	print list(resnet50.children())[-2]
    # densenet121 = torchvision.models.densenet121(pretrained = False)
    # print list(densenet121.children())[-1]