import torch
import torch.nn as nn
import convgru
import Resblock

class ResBlock_ConvGRU_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock_ConvGRU_Downsample, self).__init__()
      
        # ResBlock
        self.res = Resblock.ResBlock(in_channels, out_channels)
        
        # ConvGRU
        self.conv_gru = convgru.ConvGRU(out_channels, 3)
        
        # Downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, hidden_state = None):
        # ResBlock
        x = self.res(x)
        # ConvGRU
        h = self.conv_gru(x, hidden_state)
        # TAKES THE RESIDUAL FOR CONCATENTATION
        # Downsampling
        x = self.downsample(h)
        
        return x, h