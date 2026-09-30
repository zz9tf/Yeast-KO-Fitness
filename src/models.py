import torch
import torch.nn as nn

D = 256


def block(i, o, p):
    return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, p=0.5):
        super().__init__()
        self.net = nn.Sequential(block(in_dim, 512, p), block(512, 256, p), nn.Linear(256, D))

    def forward(self, x):
        return self.net(x)


class DotHead(nn.Module):
    def __init__(self, n_cond):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_cond, D) * 0.02)

    def forward(self, h):
        return h @ self.E.T


class FiLMHead(nn.Module):
    def __init__(self, n_cond):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_cond, D) * 0.02)
        self.Wg = nn.Linear(D, D, bias=False)
        self.Wb = nn.Linear(D, D, bias=False)
        self.v = nn.Parameter(torch.randn(D) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h):
        G, B = self.Wg(self.E), self.Wb(self.E)
        film = h @ (G * self.v).T + (B @ self.v)[None, :]
        return h @ self.E.T + self.alpha * film


class Model(nn.Module):
    def __init__(self, in_dim, n_cond, head="film"):
        super().__init__()
        self.encoder = MLPEncoder(in_dim)
        self.head = FiLMHead(n_cond) if head == "film" else DotHead(n_cond)

    def forward(self, x):
        return self.head(self.encoder(x))


def masked_mse(pred, y, m):
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def gene_pearson(pred, y, m, eps=1e-8):
    n = m.sum(1).clamp(min=2)[:, None]
    pc = (pred - (pred * m).sum(1, keepdim=True) / n) * m
    yc = (y - (y * m).sum(1, keepdim=True) / n) * m
    return (pc * yc).sum(1) / ((pc.pow(2).sum(1) * yc.pow(2).sum(1)).sqrt() + eps)
