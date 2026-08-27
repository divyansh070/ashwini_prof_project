import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss with a single-scale Gaussian (RBF) kernel.
    As formulated in HybridoNet-Adapt (Tran et al., 2025):
        k(x, y) = exp(- ||x - y||^2 / (2 * sigma^2))
        L_MMD = E[k(xs, xs')] + E[k(xt, xt')] - 2 * E[k(xs, xt)]
    """

    def __init__(self, sigma: float = 1.0, fix_sigma: bool = True):
        super().__init__()
        self.sigma = sigma
        self.fix_sigma = fix_sigma

    def _pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise squared Euclidean distances: ||x_i - y_j||^2
        """
        # x: (N, D), y: (M, D)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (M, 1)
        dist = x_norm + y_norm.t() - 2.0 * torch.matmul(x, y.t())
        return torch.clamp(dist, min=0.0)

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluates single-scale Gaussian kernel matrix between x and y.
        """
        dist = self._pairwise_dist(x, y)
        if not self.fix_sigma:
            # Median heuristic for adaptive sigma
            median_dist = torch.median(dist.detach())
            sigma_sq = 2.0 * (median_dist if median_dist > 0 else self.sigma ** 2)
        else:
            sigma_sq = 2.0 * (self.sigma ** 2)

        return torch.exp(-dist / sigma_sq)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes MMD loss between source features (N, D) and target features (M, D).
        """
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)

        n = source.size(0)
        m = target.size(0)

        if n == 0 or m == 0:
            return torch.tensor(0.0, device=source.device)

        k_ss = self._gaussian_kernel(source, source)
        k_tt = self._gaussian_kernel(target, target)
        k_st = self._gaussian_kernel(source, target)

        mmd_loss = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        return torch.clamp(mmd_loss, min=0.0)
