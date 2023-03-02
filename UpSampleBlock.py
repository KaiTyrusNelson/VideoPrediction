import torch
import torchvision
import torch.nn as nn

class ResBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlockUpsample, self).__init__()
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x, residual):
        
        residual = torchvision.transforms.functional.center_crop(residual, x.shape[2])
        out = torch.concat([x, residual], axis =1)
        
        out = self.residual_block(out)
        
        
        out = self.upsample(out)
        out = self.relu(out)
        return out