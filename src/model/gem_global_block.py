import torch


class GeMGlobalBlock(torch.nn.Module):

    def __init__(self, p: float = 3., eps: float = 1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = self.pool(x)
        x = x.pow(1.0 / self.p)
        return x.view(x.size(0), x.size(1))