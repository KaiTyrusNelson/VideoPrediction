import torch
import torch.nn as nn
import convgru

class ResBlock_ConvGRU_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock_ConvGRU_Downsample, self).__init__()
        
        self.out_channels = out_channels

        # ResBlock
        
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        # ConvGRU TO MAKE
        
        self.conv_gru = convgru.ConvGRU(out_channels, out_channels, 3)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        # Downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, hidden_state = None):
        # ResBlock
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        # ConvGRU
        
        h = self.conv_gru(x, hidden_state)
        x = self.relu3(h)
        x = self.bn3(x)
        
        # FIX THIS!!!! THIS NEEDS TO BE PASSED A HIDDEN STATE
        residual = x
        # TAKES THE RESIDUAL FOR CONCATENTATION
        # Downsampling
        x = self.downsample(x)
        
        return x, h, residual