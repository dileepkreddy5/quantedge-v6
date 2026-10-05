"""Conditional autoencoder — Gu, Kelly & Xiu (2021). Betas from a small network on the characteristics;
factors from characteristic-managed portfolios; expected return = beta(z)' λ with λ the average training factor."""
import numpy as np, torch, torch.nn as nn


class CAE(nn.Module):
    def __init__(self, L, K=3, hidden=32):
        super().__init__(); self.beta = nn.Sequential(nn.Linear(L, hidden), nn.ReLU(), nn.Linear(hidden, K)); self.fac = nn.Linear(L, K)
    def forward(self, z, X, idx): return (self.beta(z) * self.fac(X)[idx]).sum(1)


def fit_cae(Z, y, dates, K=3, epochs=25, seed=0):
    torch.manual_seed(seed); _, inv = np.unique(dates, return_inverse=True)
    X = np.zeros((inv.max() + 1, Z.shape[1]), "float32"); np.add.at(X, inv, Z * y[:, None]); X /= np.bincount(inv)[:, None]
    m = CAE(Z.shape[1], K); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    Zt, yt, Xt, it = torch.tensor(Z), torch.tensor(y, dtype=torch.float32), torch.tensor(X), torch.tensor(inv)
    for _ in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 4096):
            bb = perm[i:i + 4096]; opt.zero_grad(); ((m(Zt[bb], Xt, it[bb]) - yt[bb]) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): lam = m.fac(Xt).mean(0)
    return m, lam


def predict_cae(m, lam, Z):
    with torch.no_grad(): return (m.beta(torch.tensor(Z)) * lam).sum(1).numpy()
