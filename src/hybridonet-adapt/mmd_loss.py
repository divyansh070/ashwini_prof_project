import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class MMDLoss(nn.Module):
    """
    Multi-Kernel Maximum Mean Discrepancy (MK-MMD) Loss.

    ACCURACY FIX: the original implementation used a single, hardcoded Gaussian
    kernel with sigma=1.0, applied directly on 128-D LayerNorm'd features. A
    single fixed bandwidth on high-dimensional features is a well-known
    failure mode for MMD-based domain adaptation: if the bandwidth doesn't
    match the actual scale of pairwise distances in the batch, the kernel
    saturates near 0 or near 1 for almost all pairs, producing a near-constant,
    uninformative loss with weak gradients.

    This version instead sums several Gaussian kernels at geometrically spaced
    bandwidths around a bandwidth estimated from the data itself, following
    the standard multi-kernel MMD formulation used in domain-adaptation work
    such as DAN/JAN. This is far more robust to the actual scale of the
    features than a single fixed-sigma kernel.

        k(x, y) = sum_i exp(- ||x - y||^2 / bandwidth_i)
        L_MMD   = E[k(xs, xs')] + E[k(xt, xt')] - 2 * E[k(xs, xt)]
    """

    def __init__(
        self,
        kernel_num: int = 5,
        kernel_mul: float = 2.0,
        fix_sigma: Optional[float] = None
    ):
        """
        Args:
            kernel_num: number of Gaussian kernels to sum (multi-kernel MMD).
            kernel_mul: geometric spacing factor between kernel bandwidths.
            fix_sigma: if provided, use this as the base bandwidth (sigma^2)
                instead of estimating it from the batch. If None (default),
                the bandwidth is estimated per-batch from the mean pairwise
                squared distance, which adapts automatically to the actual
                scale of the features instead of relying on a hand-picked
                constant.
        """
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def _pairwise_sq_dist(self, total: torch.Tensor) -> torch.Tensor:
        """
        Computes the full pairwise squared Euclidean distance matrix for a
        single stacked tensor `total` of shape (N, D).
        """
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        dist = ((total0 - total1) ** 2).sum(dim=2)
        return torch.clamp(dist, min=0.0)

    def _multi_gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Builds the summed multi-kernel Gaussian matrix over the concatenation
        of source and target samples. Returns an (N+M, N+M) kernel matrix.
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        l2_distance = self._pairwise_sq_dist(total)

        if self.fix_sigma is not None:
            bandwidth = float(self.fix_sigma)
        else:
            # Adaptive bandwidth: mean pairwise squared distance over the batch
            # (the -n_samples term excludes the zero self-distance diagonal).
            denom = max(n_samples ** 2 - n_samples, 1)
            bandwidth = (l2_distance.detach().sum() / denom).item()
            if not np.isfinite(bandwidth) or bandwidth <= 0.0:
                bandwidth = 1.0

        # Center a geometric ladder of kernel_num bandwidths on the estimated/fixed bandwidth.
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_vals = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_vals) / float(self.kernel_num)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-kernel MMD loss between source features (N, D) and
        target features (M, D).
        """
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)

        n = source.size(0)
        m = target.size(0)
        if n == 0 or m == 0:
            return torch.tensor(0.0, device=source.device)

        kernels = self._multi_gaussian_kernel(source, target)
        k_ss = kernels[:n, :n]
        k_tt = kernels[n:, n:]
        k_st = kernels[:n, n:]

        mmd_loss = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        return torch.clamp(mmd_loss, min=0.0)