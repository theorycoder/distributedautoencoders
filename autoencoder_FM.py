# single-user PALM/FM (one encoder, one decoder)
# Uses FastDP: https://github.com/thecml/dpsgd-optimizer
import os
import math
import json
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fastDP import PrivacyEngine

# ----- Config -----
C = 4
n = 14
l = 7
tot_data_sz = 36480  # unchanged dataset size
num_user = 1

# privacy_mode = 'nonprivate'
# privacy_mode = 'dpsgd'
# privacy_mode = 'PALM'
privacy_mode = 'FM'

# Defaults / switches
use_custom_loss = 0
app_FM_DP = 0
dpsgd = 0
app_sen_noise = 1
sen_noise_sig = 5        # set to 1 if app_sen_noise=0 and you use DP-SGD
use_bm = 1
use_const = 1            # ignored for FM below (set to 0)

if privacy_mode == 'PALM':
    use_custom_loss = 1
    app_FM_DP = 1
elif privacy_mode == 'FM':
    use_custom_loss = 1
    app_FM_DP = 1
    use_const = 0
elif privacy_mode == 'dpsgd':
    dpsgd = 1
    use_custom_loss = 1
    app_FM_DP = 0
    if app_sen_noise == 0:
        sen_noise_sig = 1
elif privacy_mode == 'nonprivate':
    use_custom_loss = 0
    app_FM_DP = 0
else:
    raise ValueError("Invalid privacy_mode. Choose from 'nonprivate', 'PALM', 'FM', or 'dpsgd'.")

# epsilon index
try:
    I = int(sys.argv[1]) if len(sys.argv) > 1 else 0
except ValueError:
    print("Invalid input. Using default I=0")
    I = 0

eps = [0.01, 0.1, 1.0, 10.0, 20.0]
I = max(0, min(I, len(eps)-1))  # clamp to valid range

# ----- Model -----
class DA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(n, l)
        self.decoder = nn.Linear(l, n)

    def forward(self, x):
        e = torch.sigmoid(self.encoder(x))     # [B, l]
        out = torch.sigmoid(self.decoder(e))   # [B, n]
        return out, e

# ----- Loss (NaN-safe) -----
def my_loss(W_dec, e, y_true, y_pred, W_enc):
    device = y_pred.device
    # constant shift for PALM (disabled for FM via use_const=0 above)
    const_val = use_const * float(os.getenv("DECODER_CONST", "2.5"))

    # shift decoder weights
    W_dec = W_dec + const_val  # [l, n]

    # Compute bm with stabilizations
    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)   # [n]
    WN = torch.matmul(W_enc, N)                                    # [l]
    WN = torch.clamp(WN, min=-20.0, max=20.0)                      # avoid exp overflow
    exp_WN = torch.exp(WN)                                         # [l]

    e_s = e.squeeze().clamp(1e-6, 1.0 - 1e-6)                      # stabilize denom
    denom = 1 + (exp_WN - 1) * e_s
    bm_vec = exp_WN / denom
    bm_vec = torch.nan_to_num(bm_vec, nan=0.0, posinf=1e6, neginf=0.0)
    bm_val = bm_vec.sum().clamp(1e-6, 1e6).item()                  # bounded float

    # Scaling for FM/PALM (positive float)
    scale_FM = float((1.5 * n) / (math.sqrt(2.0) * eps[I]))
    if app_sen_noise and use_bm:
        scale_FM *= bm_val
    if const_val > 0:
        eb = e * const_val
        cj = torch.sum(eb).item()
        scale_FM += float(n * cj * (0.5 * cj + 2.0))
    # final guard
    scale_FM = max(1e-8, float(scale_FM))

    # Shared terms
    alpha_ji1 = math.log(2.0)
    alpha_ji2 = 0.5 - y_pred.squeeze()
    alpha_ji3 = 0.5 * y_pred.squeeze() - 0.25

    # Single-user path
    a = torch.matmul(e, W_dec[0:l, :])  # [B, n]
    b = a ** 2

    if app_FM_DP:
        # construct Laplace on correct device/dtype
        dist = torch.distributions.Laplace(
            loc=torch.tensor(0.0, device=device, dtype=a.dtype),
            scale=torch.tensor(scale_FM, device=device, dtype=a.dtype),
        )
        noise = dist.sample(a.shape[-1:])  # [n]
        if app_sen_noise:
            alpha_ji2 = alpha_ji2 * bm_val
            alpha_ji3 = alpha_ji3 * bm_val
        s = torch.sum(alpha_ji1 + (alpha_ji2 + noise) * a + (alpha_ji3 + noise) * b)
    elif dpsgd == 1:
        alpha_ji2 = alpha_ji2 * bm_val
        alpha_ji3 = alpha_ji3 * bm_val
        s = torch.sum(alpha_ji1 + alpha_ji2 * a + alpha_ji3 * b)
    else:
        s = torch.sum(alpha_ji1 + alpha_ji2 * a + alpha_ji3 * b)

    return s

# ----- Data -----
with open('fitbit_dataset_expanded_80x.json') as f:
    inp = np.array(json.load(f), dtype=np.float32)

inp = np.reshape(inp, (tot_data_sz, n)).astype(np.float32)
for j in range(n):
    mx = float(np.max(inp[:, j]))
    if mx > 0:
        inp[:, j] /= mx

X = inp.copy()
noise_gauss = np.random.normal(0, sen_noise_sig, (tot_data_sz, n)).astype(np.float32)
X_hat = X + noise_gauss if app_sen_noise else X

X = torch.tensor(X, dtype=torch.float32)
X_hat = torch.tensor(X_hat, dtype=torch.float32)

dataset = TensorDataset(X, X_hat)
train_size = int(0.8 * tot_data_sz)
test_size = tot_data_sz - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

# ----- bm estimation (same approach as original, but safe) -----
def findbm(model, x, sen_noise_sig, n):
    device = x.device
    W_enc = model.encoder.weight  # [l, n]
    e = torch.sigmoid(model.encoder(x))  # [l]
    # Stabilized bm for a single sample
    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)  # [n]
    WN = torch.tensordot(W_enc.T, N, dims=([0], [0]))            # [l]
    bm_sum = 0.0
    for k in range(WN.numel()):
        v = float(WN[k].item())
        v = max(-20.0, min(20.0, v))
        exp_v = math.exp(v)
        e_k = float(e[k].item())
        e_k = min(max(e_k, 1e-6), 1.0 - 1e-6)
        denom = 1.0 + (exp_v - 1.0) * e_k
        term = exp_v / denom if denom != 0.0 else 0.0
        bm_sum += term
    bm_sum = min(max(bm_sum, 1e-6), 1e6)
    return bm_sum

# ----- Train / Eval -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DA().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# estimate bm once
with torch.no_grad():
    bm_values = []
    for i in range(min(X.size(0), 2048)):  # cap to speed up
        bm_values.append(findbm(model, X[i].to(device), sen_noise_sig, n))
    bm = float(np.mean(bm_values)) if bm_values else 1.0
    bm = min(max(bm, 1e-6), 1e6)

# FastDP PrivacyEngine setup
if dpsgd == 1:
    scale_dpsgd = (n + 2 * n * C) / (2.0 * eps[I])
    if use_bm == 1:
        scale_dpsgd = (n * bm + 2 * n * C) / (2.0 * eps[I])
    privacy_engine = PrivacyEngine(
        model,
        batch_size=1,
        sample_size=len(train_set),
        epochs=10,
        target_epsilon=eps[I],
        noise_multiplier=scale_dpsgd,
        clipping_fn='Abadi',
        clipping_mode='ghost',
        clipping_style='all-layer',
        max_grad_norm=C,
    )
else:
    # non-private w.r.t. grad noise; FM/PALM noise is inside the custom loss
    privacy_engine = PrivacyEngine(
        model,
        batch_size=1,
        sample_size=len(train_set),
        epochs=10,
        target_epsilon=None,
        noise_multiplier=0.0,
        clipping_fn='none',
        clipping_mode='none',
        origin_params=None,
    )
privacy_engine.attach(optimizer)

start_time = time.time()

for epoch in range(1):
    model.train()
    for x, x_hat in train_loader:
        x, x_hat = x.to(device), x_hat.to(device)
        optimizer.zero_grad()

        y_pred, e = model(x)
        W_dec = model.decoder.weight.T  # [l, n]
        W_enc = model.encoder.weight    # [l, n]

        if use_custom_loss:
            loss = my_loss(W_dec, e, x_hat, y_pred, W_enc)
        else:
            # y_pred already passed through sigmoid → use BCE (not BCEWithLogits)
            loss = F.binary_cross_entropy(y_pred, x_hat)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}")

tot_time = time.time() - start_time
print(f"Total simulation time: {tot_time:.2f} seconds")

# ----- Evaluation -----
model.eval()
with torch.no_grad():
    X_pred, _ = model(X_hat.to(device))
mse = torch.mean((X_pred.cpu() - X) ** 2).item()
acc = 1.0 - mse
print(f"Accuracy (1 - MSE): {acc:.6f}")

# ----- Logs -----
if dpsgd == 1:
    filename = f"dpsgdaccuracy_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif app_sen_noise == 1 and privacy_mode == 'PALM':
    filename = f"PALMaccuracy_noisyInp_{I}_{use_bm}_{sen_noise_sig}.txt"
elif app_sen_noise == 0 and use_custom_loss == 1 and app_FM_DP == 1 and privacy_mode == 'PALM':
    filename = f"PALMaccuracy_noislessInp_{I}.txt"
elif app_sen_noise == 1 and privacy_mode == 'FM':
    filename = f"FMaccuracy_noisyInp_{I}_{use_bm}_{sen_noise_sig}.txt"
elif app_sen_noise == 0 and use_custom_loss == 1 and app_FM_DP == 1 and privacy_mode == 'FM':
    filename = f"FMaccuracy_noislessInp_{I}.txt"
elif privacy_mode == 'nonprivate':
    filename = f"nonprivate_{I}.txt"
with open(filename, 'a') as f:
    f.write(str(acc) + " ")

if privacy_mode == 'nonprivate':
    filename = f"nonPrivate_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode == 'dpsgd':
    filename = f"dpsgd_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode == 'PALM':
    filename = f"PALM_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode == 'FM':
    filename = f"fm_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
with open(filename, 'a') as f:
    f.write(str(tot_time) + " ")

