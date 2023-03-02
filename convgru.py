import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super().__init__()
        
        #Passed the input channels, and hidden channels, as well as the kernel size
        # To do, make map to same dim regardless of kernel size
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.W_r = nn.Sequential(
                   nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size, padding = 1),
                   nn.Sigmoid())
        self.W_u = nn.Sequential(
                   nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding = 1),
                   nn.Sigmoid())
        self.W_c = nn.Sequential(
                   nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding = 1),
                   nn.ReLU())
        
        
    def forward(self, x, h = None):
        
        if (h == None):
            h = torch.zeros(x.shape[0], self.hidden_size, x.shape[2], x.shape[3])
        
        concat = torch.concat([x,h], axis = 1)
        
        # old hidden state
        r = self.W_r(concat)
        u = self.W_u(concat)
        
        concat2 = torch.concat([x,torch.mul(r, u)], axis = 1)
        
        c = self.W_c(concat2)
        
        # new hidden state
        h_new = torch.mul(u, h) + torch.mul(1-u, c)
        
        
        
        return h_new