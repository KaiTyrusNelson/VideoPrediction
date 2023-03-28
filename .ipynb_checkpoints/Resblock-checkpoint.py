import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3 , padding = 1):
        super(ResBlock, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # NON-LINEAR FORWARD METHOD
            self.Fw = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size= kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
            
            nn.Conv2d(output_channels, output_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(output_channels),
        )
        
        #LIEAR PROJECTION TO CONNECT RESIDUAL TO THE NON-LINEARITY
        self.lin_proj = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = padding)

        
    def forward(self, x):
        
        
        x_new = self.Fw(x)
        
        x_new = x_new + self.lin_proj(x)
        
        return x_new
 