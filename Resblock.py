import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3 , padding = 1):
        super(ResBlock, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.Fw = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size= kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
            
            nn.Conv2d(output_channels, output_channels, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
        )

        
    def forward(self, x):
        
        
        x_new = self.Fw(x)
        
        x_new = self.Fw(x) + torch.concat([x, torch.zeros(x.shape[0], self.output_channels - self.input_channels, x.shape[2], x.shape[3])], axis =  1)
        
        return x_new
 