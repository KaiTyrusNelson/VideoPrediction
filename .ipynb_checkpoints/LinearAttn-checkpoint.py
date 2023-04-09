import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(self, head_size, n_embed):
        super().__init__()
        self.key = torch.nn.Linear(n_embed, head_size, bias = False)
        self.query = torch.nn.Linear(n_embed, head_size, bias = False)
        self.value = torch.nn.Linear(n_embed, head_size, bias = False)
        self.head_size = head_size
        
    def forward(self, x):

        k = self.key(x)
        q = self.query(x)

        wei = q @k.transpose(-2, -1) / self.head_size**0.5
        wei = torch.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        
    def forward(self, x):
        
        B, C, H, W = x.shape
        
        print(x.shape)
        
        x = x.view(B, H*W, C)
        
        print(x.shape)
        
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.proj(out)
        
        return out.view(B, C, H, W)
