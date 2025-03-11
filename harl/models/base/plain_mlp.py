import torch.nn as nn
from harl.utils.models_tools import get_active_func


class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)] # get_active_func를 통해 nn.ReLU()를 받아옴
        self.mlp = nn.Sequential(*layers)   # layers 요소들을 순차적으로 전달하기 위해 *을 붙임.

    def forward(self, x):
        return self.mlp(x)
