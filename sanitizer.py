import math
import torch
import numpy as np

class AmortizedGaussianSanitizerPT:
    """
    PyTorch replacement sanitizer.
    - Clips x to l2norm_bound (if provided), otherwise uses default_l2norm_bound.
    - Adds Gaussian noise with std = sigma * l2norm_bound (if sigma provided),
      otherwise computes sigma from (eps, delta) via: sigma = sqrt(2 ln(1.25/delta)) / eps
      (then effective noise std = sigma * l2norm_bound).
    - Returns a detached tensor on the same device/dtype as x.
    """

    def __init__(self, accountant=None, default_l2norm_bound=1.0, do_clip=True):
        self.accountant = accountant
        self.default_l2norm_bound = float(default_l2norm_bound)
        self.do_clip = bool(do_clip)

    def _compute_sigma_from_eps_delta(self, eps_delta):
        eps, delta = eps_delta
        if eps <= 0 or delta <= 0:
            raise ValueError("eps and delta must be > 0")
        # Gaussian mechanism formula used in many references
        sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        return sigma

    def sanitize(self, x, eps_delta=None, sigma=None, l2norm_bound=None,
                 add_noise=True, num_examples=None, tensor_name=None):
        """
        x: PyTorch tensor (gradient tensor)
        eps_delta: tuple (eps, delta) or None. Used only if sigma is None.
        sigma: if provided, interpreted as the Gaussian sigma factor (not yet multiplied by l2 bound).
               Final noise std = sigma * l2norm_bound.
        l2norm_bound: clipping norm. If None uses default_l2norm_bound.
        add_noise: if False, only clipping and reduction is applied (no noise).
        Returns: sanitized tensor (detached), same shape as x.
        """
        # ensure tensor
        if not torch.is_tensor(x):
            raise TypeError("Expected torch.Tensor for x")

        device = x.device
        dtype = x.dtype

        if l2norm_bound is None:
            l2norm_bound = float(self.default_l2norm_bound)
        else:
            l2norm_bound = float(l2norm_bound)

        # compute sigma if not given
        if sigma is None:
            if eps_delta is None:
                raise ValueError("Either sigma or eps_delta must be provided")
            sigma = self._compute_sigma_from_eps_delta(eps_delta)
        sigma = float(sigma)

        # Clip: for per-parameter gradients we clip the whole tensor x by its L2 norm.
        if self.do_clip:
            # compute norm of x (L2)
            x_norm = torch.norm(x).item()
            if x_norm > l2norm_bound and x_norm > 0:
                x = (x / (x_norm + 1e-16)) * l2norm_bound
            # else leave x unchanged
        # If not clipping, we still may want to track norm for accounting

        # Optionally accumulate privacy via accountant (if provided)
        if self.accountant is not None and eps_delta is not None:
            # account for one "example" per sanitize call; adapt if you are using amortized accounting
            # accountant.accumulate_privacy_spending(eps_delta, sigma, num_examples)
            try:
                self.accountant.accumulate_privacy_spending(eps_delta, sigma, num_examples)
            except Exception:
                # if your accountant API differs, adapt this call
                pass

        if add_noise:
            # noise std scaled by clipping bound (same style as original code)
            noise_std = sigma * l2norm_bound
            if noise_std > 0:
                noise = torch.normal(mean=0.0, std=noise_std, size=x.shape, device=device, dtype=dtype)
            else:
                noise = torch.zeros_like(x)
            saned = x + noise
        else:
            # if no noise, just return clipped (or original) gradient
            saned = x

        # Detach and return value (no grad)
        return saned.detach()

