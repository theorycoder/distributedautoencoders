#two encoders, 1 decoder
#This program uses the FastDP library available at https://github.com/thecml/dpsgd-optimizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
import json
import sys
from torch.utils.data import DataLoader, TensorDataset
from fastDP import PrivacyEngine
import time

from accountant import SimpleAccountant
from sanitizer_pt import AmortizedLaplaceSanitizerPT
from pathlib import Path
import collections
EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

C = 4
#privacy_mode = 'nonprivate'  # ← CHANGE this to 'DFM' or 'dpsgd' as needed
#privacy_mode = 'dpsgd' 
#privacy_mode = 'DFM' 
privacy_mode = 'FM' 
num_user = 1

if privacy_mode == 'FM': 
    num_user = 1

# Defaults
use_custom_loss = 0
app_FM_DP = 0
dpsgd = 0
app_sen_noise = 1
sen_noise_sig = 5 #make this 5 even if app_sen_noise=0 
use_bm = 1
use_const=0

if privacy_mode == 'DFM':
    use_custom_loss = 1
    app_FM_DP = 1
elif privacy_mode == 'FM':
    use_custom_loss = 1
    app_FM_DP = 1
    use_const=0
elif privacy_mode == 'dpsgd':
    dpsgd = 1
    use_custom_loss = 1
    app_FM_DP = 0
    if app_sen_noise==0:
        sen_noise_sig = 1
elif privacy_mode == 'nonprivate':
    use_custom_loss = 1
    app_FM_DP = 0
else:
    raise ValueError("Invalid privacy_mode. Choose from 'nonprivate', 'DFM', or 'dpsgd'.")

use_bm=1
try:
    #I = int(input("Privacy budget index (0-4): ") or "0")  # command line input
    I = int(sys.argv[1]) if len(sys.argv) > 1 else 0 #for one thread use command: for r in {1..14}; do for i in {0..4}; do python3 autoencoder2_pytorch.py $i; done; done   use run.sh  
    #I=0 #manually pick one values of I using command: for i in {1..14}; do python3 autoencoder2_pytorch.py; done
    #I = int(os.getenv("EPS_INDEX", "0")) #run using weight constant sweep
except ValueError:
    print("Invalid input. Using default I=0")
    I = 0

#gradlog_filename = f"gradlog_I{I}.txt"

#eps = [0.000001]
#eps = [0.01, 0.1, 1.0, 10.0, 20] 
eps = [0.1, 0.2, 0.4, 0.8, 1.6] 
        
        
class DA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(n, l)
        self.decoder = nn.Linear(num_user * l, n)

    def forward(self, x):

        user_inputs = torch.chunk(x, num_user, dim=0)

        encodings = []

        for ui in user_inputs:
            e = torch.sigmoid(self.encoder(ui))
            encodings.append(e)

        dec_input = torch.cat(encodings, dim=1)

        out = torch.sigmoid(self.decoder(dec_input))

        return out, encodings  
        

def my_loss(W_dec, encodings, y_true, y_pred, W_enc):
    device = y_pred.device
    const_val = use_const * float(os.getenv("DECODER_CONST", "2.5"))
    W_dec = W_dec + const_val

    # --- Compute bm using first encoding (same behavior as before) ---
    e_ref = encodings[0]

    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)
    WN = torch.matmul(W_enc, N)
    WN = torch.clamp(WN, -20, 20)   # prevent overflow
    exp_WN = torch.exp(WN)
    bm_vec = exp_WN / (1 + (exp_WN - 1) * e_ref.squeeze())
    bm = bm_vec.sum()

    # --- Scaling ---
    scale_FM = (1.5 * n) / (math.sqrt(2) * eps[I])
    if app_sen_noise and use_bm:
        scale_FM *= bm

    if const_val > 0:
        e1b = e_ref * const_val
        cj = torch.sum(e1b)
        scale_FM += n * cj * (0.5 * cj + 2) / (math.sqrt(2) * eps[I])

    # --- Shared terms ---
    alpha_ji1 = math.log(2)
    alpha_ji2 = 0.5 - y_true.squeeze()
    alpha_ji3 = 0.5 * y_true.squeeze() - 0.25

    total_sum = 0

    # --- Loop over all users ---
    for i, e in enumerate(encodings):

        start = i * l
        end = (i + 1) * l

        a = torch.matmul(e, W_dec[start:end, :])
        b = a ** 2

        if app_FM_DP:
            noise = torch.distributions.Laplace(0, scale_FM).sample([n]).to(device)

            if app_sen_noise:
                alpha2 = alpha_ji2 * bm
                alpha3 = alpha_ji3 * bm
            else:
                alpha2 = alpha_ji2
                alpha3 = alpha_ji3

            term = torch.sum(alpha_ji1 + (alpha2 + noise) * a + (alpha3 + noise) * b)

        elif dpsgd == 1:
            alpha2 = alpha_ji2 * bm
            alpha3 = alpha_ji3 * bm
            term = torch.sum(alpha_ji1 + alpha2 * a + alpha3 * b)

        else:
            term = torch.sum(alpha_ji1 + alpha_ji2 * a + alpha_ji3 * b)

        total_sum += term

    return total_sum


#with open('fitbit_dataset.json') as f:
#with open('fitbit_dataset_expanded_20x.json') as f:

'''
with open('fitbit_dataset_expanded_80x.json') as f:
    inp = np.array(json.load(f))
inp = np.reshape(inp, (-1, n)).astype(np.float32)
for i in range(n):
    inp[:, i] /= np.max(inp[:, i])
tot_data_sz = inp.shape[0]
'''








# -----------------------------
# Parameters
# -----------------------------
latent_ratio = 0.5
noise_std = 0.01
num_copies = 1
base_features = 14
tot_data_sz = 9120
#tot_data_sz = 36480   # FIXED dataset size

# -----------------------------
# Load dataset
# -----------------------------
with open('fitbit_dataset_expanded_20x.json') as f:
#with open('fitbit_dataset_expanded_80x.json') as f:
    inp = np.array(json.load(f)).astype(np.float32)

# reshape to original features
inp = np.reshape(inp, (-1, base_features))

# -----------------------------
# Expand feature dimension
# -----------------------------
inp = np.tile(inp, (1, num_copies))

noise = np.random.normal(0, noise_std, inp.shape).astype(np.float32)
inp += noise

# -----------------------------
# Keep dataset size fixed
# -----------------------------
inp = inp[:tot_data_sz]

# -----------------------------
# Update dimensions
# -----------------------------
n = inp.shape[1]
l = max(1, int(n * latent_ratio))

print(f"tot_data_sz = {tot_data_sz}, n = {n}, l = {l}, num_user = {num_user}")
print(f"samples per encoder = {tot_data_sz/num_user}")

# -----------------------------
# Normalize features
# -----------------------------
for i in range(n):
    inp[:, i] /= (np.max(inp[:, i]) + 1e-8)

# Convert to float32 for PyTorch
inp = inp.astype(np.float32)


X = inp.copy()
noise1 = np.random.laplace(0, 1, (tot_data_sz, n)).astype(np.float32)
noise2 = np.random.normal(0, sen_noise_sig, (tot_data_sz, n)).astype(np.float32)

'''
for i in range(tot_data_sz):
    for j in range(n):
        if random.randint(0, 100) >= 90:
            X[i, j] = 1.0
'''

X_hat = X + noise2 if app_sen_noise else X
X = torch.tensor(X, dtype=torch.float32)
X_hat = torch.tensor(X_hat, dtype=torch.float32)

dataset = TensorDataset(X, X_hat)
train_size = int(0.8 * tot_data_sz)
test_size = tot_data_sz - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=num_user, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=num_user, shuffle=True, drop_last=True)


def findbm(model, x, sen_noise_sig, n):
    device = x.device
    W_dec = model.decoder.weight.T  # shape [2l, n]
    W_enc1 = model.encoder.weight  # shape [l, n]
    e1 = torch.sigmoid(model.encoder(x))  # shape [l]
    e1 = e1.unsqueeze(0)  # Adds an extra dimension, making it [1, l]
    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)  # shape [n]
    WN = torch.tensordot(W_enc1.T, N, dims=([0], [0]))  # shape [l]
    bm = 0.0
    for i in range(len(WN)):
        exp_WN_i = math.exp(WN[i].item())
        enc_val = e1[0, i].item()  # Now e1 has shape [1, l]
        b1i = exp_WN_i / (1 + (exp_WN_i - 1) * enc_val)
        bm += b1i
    return bm



# ----- Training -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DA().to(device)
optimizer = optim.Adam(model.parameters())
rnidx = torch.randint(n, (1,))
bm_values = []

for i in range(X.size(0)):  # X.size(0) gives the number of samples
    sample = X[i]  # Extract the i-th sample
    bm = findbm(model, sample, sen_noise_sig, n)  # Compute bm for the sample
    bm_values.append(bm)
bm_tensor = torch.tensor(bm_values)
bm = bm_tensor.mean()
#print('bm: ',bm)

        
'''    
if dpsgd==1:
    scale_dpsgd=(n+2*n*C)/(2*eps[I]) #division by 2 due to average of 1 and 2C
    if use_bm==1:
        scale_dpsgd=(n*bm+2*n*C)/(2*eps[I])    
    privacy_engine = PrivacyEngine(model, batch_size=1, sample_size=len(train_set), epochs=10, target_epsilon=eps[I], noise_multiplier=scale_dpsgd, clipping_fn='Abadi', clipping_mode='ghost', clipping_style='all-layer',
    max_grad_norm=C)
else:
    privacy_engine = PrivacyEngine(model, batch_size=1, sample_size=len(train_set), epochs=10, target_epsilon=None, noise_multiplier=0.0, clipping_fn='none', clipping_mode='none', origin_params=None,) #non-private as FM noise in injected into loss function
privacy_engine.attach(optimizer) #Laplace noise is added in fast-differential-privacy/fastDP/supported_layers_grad_samplers.py which uses noise_multiplier as the s.d.
'''

accountant = SimpleAccountant()
sanitizer = AmortizedLaplaceSanitizerPT(accountant, [C / num_user, True])

start_time = time.time()

for epoch in range(10):
    model.train()
    train_loss = 0.0
    
    for x, x_hat in train_loader:
        x, x_hat = x.to(device), x_hat.to(device)
        optimizer.zero_grad()

        # forward pass
        y_pred, encodings = model(x)

        W_dec = model.decoder.weight.T
        W_enc = model.encoder.weight

        if use_custom_loss:
            loss = my_loss(W_dec, encodings, x_hat, y_pred, W_enc)
        else:
            #loss = F.binary_cross_entropy_with_logits(y_pred, x_hat)
            loss = F.binary_cross_entropy_with_logits(y_pred.repeat(x_hat.size(0),1), x_hat)

        loss.backward()

        if dpsgd:

            agg_grads = [torch.zeros_like(p) for p in model.parameters()]

            user_targets = torch.chunk(x_hat, num_user)

            for j in range(num_user):

                optimizer.zero_grad()

                y_pred_j, encodings_j = model(x)

                if use_custom_loss:
                    loss_j = my_loss(W_dec, encodings_j, x_hat, y_pred_j, W_enc)
                else:
                    loss_j = F.binary_cross_entropy_with_logits(y_pred_j, x_hat)

                loss_j = loss_j / num_user
                loss_j.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), C)

                for i, p in enumerate(model.parameters()):
                    if p.grad is not None:
                        agg_grads[i] += p.grad.detach() / num_user

            # add DP noise
            eps_delta = EpsDelta(eps[I], 0.00001)

            if use_bm == 1:
                scale_dpsgd = (n * bm + 2 * n * C) / (2 * eps[I])
            else:
                scale_dpsgd = (n + 2 * n * C) / (2 * eps[I])

            for param, g in zip(model.parameters(), agg_grads):
                sanitized = sanitizer.sanitize(
                g,eps_delta, noise_scale=scale_dpsgd)

                param.grad = sanitized.detach()

        optimizer.step()

    print(f"Epoch {epoch+1}")
    
end_time = time.time()
tot_time=end_time - start_time
print(f"Total simulation time: {tot_time:.2f} seconds")


# ----- Evaluation -----
#model.eval()
#with torch.no_grad():
#    X_pred, _, _ = model(X_hat)  # forward pass
#    #X_pred, _, _ = model(X_hat.view(1, -1))
    
# Compute MSE and accuracy
#mse = torch.mean((X_pred - X) ** 2).item()

model.eval()
pred_list = []
true_list = []
with torch.no_grad():
    for x, x_hat in test_loader:
        x = x.to(device)
        x_hat = x_hat.to(device)

        y_pred, _ = model(x)

        pred_list.append(y_pred.cpu())
        true_list.append(x.cpu()[0:1])  # one reconstructed sample per group

X_pred = torch.cat(pred_list, dim=0)
X_true = torch.cat(true_list, dim=0)
mse = torch.mean((X_pred - X_true) ** 2).item()
acc = 1 - mse
print(f"Accuracy (1 - MSE): {acc:.6f}")

if dpsgd == 1:
    filename = f"dpsgdaccuracy_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif app_sen_noise==1 and privacy_mode=='DFM':
    filename = f"DFMaccuracy_noisyInp_{I}_{use_bm}_{sen_noise_sig}.txt"
elif app_sen_noise == 0 and use_custom_loss == 1 and app_FM_DP == 1 and privacy_mode=='DFM':
    filename = f"DFMaccuracy_noislessInp_{I}.txt"
elif app_sen_noise==1 and privacy_mode=='FM':
    filename = f"FMaccuracy_noisyInp_{I}_{use_bm}_{sen_noise_sig}.txt"
elif app_sen_noise == 0 and use_custom_loss == 1 and app_FM_DP == 1 and privacy_mode=='FM':
    filename = f"FMaccuracy_noislessInp_{I}.txt"
elif privacy_mode=='nonprivate':
    filename = f"nonprivate_{I}.txt"
with open(filename, 'a') as f:
    f.write(str(acc) + " ")

if privacy_mode=='nonprivate':
    filename = f"nonPrivate_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode=='dpsgd':
    filename = f"dpsgd_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode=='DFM':
    filename = f"DFM_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode=='FM':
    filename = f"FM_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
with open(filename, 'a') as f:
    f.write(str(tot_time) + " ")
