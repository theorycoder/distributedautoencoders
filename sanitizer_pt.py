
import torch
from collections import namedtuple

ClipOption = namedtuple("ClipOption", ["l2norm_bound", "clip"])

'''
class AmortizedLaplaceSanitizerPT:
    """
    Sanitizes gradients for DP-SGD using Laplace noise.
    """

    def __init__(self, accountant, default_option):
        """
        Args:
            accountant: an instance of SimpleAccountant (tracks privacy spending)
            default_option: [l2norm_bound, clip] e.g., [C / batch_size, True]
        """
        self.accountant = accountant
        self.default_option = ClipOption(*default_option)

    def sanitize(self, grad, eps_delta, scale, option=None, add_noise=True):
        """
        Args:
            grad: torch.Tensor — gradient to sanitize
            eps_delta: (epsilon, delta) — note: delta ignored for Laplace
            scale: sensitivity multiplier (usually scale_dpsgd)
            option: ClipOption(l2norm_bound, clip)
            add_noise: bool — whether to add Laplace noise
        """
        if option is None:
            option = self.default_option
        l2norm_bound, clip = option

        # --- Step 1: Clip gradient ---
        if clip:
            grad_norm = grad.norm(2)
            clip_coef = l2norm_bound / (grad_norm + 1e-6)
            clip_coef = min(1.0, clip_coef)
            grad = grad * clip_coef

        # --- Step 2: Add Laplace noise ---
        if add_noise:
            eps, _ = eps_delta  # delta ignored for Laplace
            laplace_scale = scale / eps  # <--- Changed from Gaussian formula to Laplace
            noise = torch.distributions.Laplace(0, laplace_scale).sample(grad.shape).to(grad.device)
            grad = grad + noise

        # --- Step 3: Track privacy spending ---
        self.accountant.accumulate_privacy_spending(eps_delta, num_examples=1)

        return grad
'''        
    

class AmortizedLaplaceSanitizerPT:

    def __init__(self, accountant, default_option):
        self.accountant = accountant
        self.default_option = ClipOption(*default_option)

    def sanitize(self, grad, eps_delta, option=None, add_noise=True, noise_scale=None):

        if option is None:
            option = self.default_option

        C, clip = option
        eps, _ = eps_delta

        # --- Clip gradient ---
        if clip:
            grad_norm = grad.norm(2)
            clip_coef = min(1.0, C / (grad_norm + 1e-12))
            grad = grad * clip_coef

        # --- Add Laplace noise ---
        if add_noise:

            if noise_scale is None:
                laplace_scale = C / (eps + 1e-12)
            else:
                laplace_scale = noise_scale

            noise = torch.distributions.Laplace(
                0, laplace_scale
            ).sample(grad.shape).to(grad.device)

            grad = grad + noise

        return grad



