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
        
        self.W_z = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
        self.U_z = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
                   
        self.W_r = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
        self.U_r = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
        
        self.W = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
        self.U = nn.Conv2d(input_size, input_size, kernel_size, padding = 1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, h = None):
        
        # create a hideen state like the input
        if (h == None):
            h = torch.zeros_like(x)
        
        z = self.sigmoid(self.W_z(x) +self.U_z(h))
        r = self.sigmoid(self.W_r(x) + self.U_r(h))
        h_candidate = self.tanh(self.W(x) + self.U(torch.mul(r, h)))
        
        # new hidden state
        h_new = torch.mul(1-z, h) + torch.mul(z, h_candidate)
        
        return h_new