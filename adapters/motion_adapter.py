import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x)) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.linear = linear_layer
        self.lora = LoRALayer(self.in_features, self.out_features, rank=rank)

    def forward(self, x):
        return self.lienar(x) + self.lora(x)
