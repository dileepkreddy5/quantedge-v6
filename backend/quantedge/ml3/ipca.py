"""IPCA — Kelly, Pruitt & Su (2019). Returns depend on K latent factors whose loadings are linear in the
characteristics: r = z' Γ f. Alternating least squares on per-month moments; expected return = z' Γ λ,
where λ is the average factor return over the training period (no future information)."""
import numpy as np


def fit_ipca(Z, y, dates, K=3, iters=30):
    _, start = np.unique(dates, return_index=True); b = list(start) + [len(dates)]
    ZZ = np.array([Z[i:j].T @ Z[i:j] / (j - i) for i, j in zip(b[:-1], b[1:])]); Zr = np.array([Z[i:j].T @ y[i:j] / (j - i) for i, j in zip(b[:-1], b[1:])])
    L = Z.shape[1]; G = np.linalg.svd(Zr.T, full_matrices=False)[0][:, :K]
    def factors(G): return np.array([np.linalg.solve(G.T @ ZZ[t] @ G + 1e-8 * np.eye(K), G.T @ Zr[t]) for t in range(len(ZZ))])
    for _ in range(iters):
        F = factors(G); A = np.zeros((L * K, L * K)); v = np.zeros(L * K)
        for t in range(len(ZZ)): A += np.kron(ZZ[t], np.outer(F[t], F[t])); v += np.kron(Zr[t], F[t])
        G = np.linalg.qr(np.linalg.solve(A + 1e-6 * np.eye(L * K), v).reshape(L, K))[0]
    return G, factors(G).mean(axis=0)
