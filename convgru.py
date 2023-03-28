import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRU(nn.Module):
    def __init__(self, input_size, kernel_size = 3):
        super().__init__()
        # arxiv 1511.06432
        # Input size == hidden size
        self.input_size = input_size
        
        self.W_z = nn.Conv2d(2*input_size, input_size, kernel_size, padding = 1)
                   
        self.W_r = nn.Conv2d(2*input_size, input_size, kernel_size, padding = 1)
        
        self.W = nn.Conv2d(2*input_size, input_size, kernel_size, padding = 1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, h = None):
        
        # create a hideen state like the input
        if (h == None):
            h = torch.zeros_like(x)
        
        
        r = self.sigmoid(self.W_r(torch.concat([x,h], axis = 1)))
        z = self.sigmoid(self.W_z(torch.concat([x,h], axis = 1)))
        
        h_candidate = self.tanh(self.W(torch.concat([x, torch.mul(r, h)], axis = 1)))
        
        # new hidden state
        h_new = torch.mul(1-z, h) + torch.mul(z, h_candidate)
        
        return h_new