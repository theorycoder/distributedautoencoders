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

C = 4
n = 14
l = 7
tot_data_sz = 456
num_user = 2
privacy_mode = 'nonprivate'  # ← CHANGE this to 'fm' or 'dpsgd' as needed
#privacy_mode = 'dpsgd' 
#privacy_mode = 'fm' 

# Defaults
use_custom_loss = 0
app_FM_DP = 0
dpsgd = 0
app_sen_noise = 0
sen_noise_sig = 1 #make this 1 for DP-SGD if app_sen_noise=0 
use_bm = 1

if privacy_mode == 'fm':
    use_custom_loss = 1
    app_FM_DP = 1
elif privacy_mode == 'dpsgd':
    dpsgd = 1
    use_custom_loss = 1
    app_FM_DP = 0
    if app_sen_noise==0:
        sen_noise_sig = 1
elif privacy_mode == 'nonprivate':
    use_custom_loss = 0
    app_FM_DP = 0
else:
    raise ValueError("Invalid privacy_mode. Choose from 'nonprivate', 'fm', or 'dpsgd'.")

use_bm=1
try:
    #I = int(input("Privacy budget index (0-6): ") or "0")  # command line input
    I = int(sys.argv[1]) if len(sys.argv) > 1 else 0 #for one thread use command: for r in {1..14}; do for i in {0..4}; do python3 autoencoder2_pytorch.py $i; done; done   use run.sh  
    #I=0 #manually pick one values of I using command: for i in {1..14}; do python3 autoencoder2_pytorch.py; done
    #I = int(os.getenv("EPS_INDEX", "0")) #run using weight constant sweep
except ValueError:
    print("Invalid input. Using default I=0")
    I = 0

#gradlog_filename = f"gradlog_I{I}.txt"

#eps = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
eps = [0.001, 0.01, 0.1, 1.0, 10.0] #0.0018, 0.0032, 0.0056, 


class DA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(n, l)
        self.encoder2 = nn.Linear(n, l)
        self.decoder = nn.Linear(num_user * l, n)

    def forward(self, x):
        e1 = torch.sigmoid(self.encoder1(x))
        e2 = torch.sigmoid(self.encoder2(x))
        dec_input = torch.cat([e1, e2], dim=1)
        out = torch.sigmoid(self.decoder(dec_input))
        return out, e1, e2
        

def my_loss(W_dec, e1, e2, y_true, y_pred, W_enc1):
    device = y_pred.device
    const_val = float(os.getenv("DECODER_CONST", "2.5"))
    #if app_FM_DP:
    W_dec = W_dec + const_val
    #W_dec = W_dec + 2.5 #improves SNR of loss function coefficients

    # --- Compute bm ---
    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)
    WN = torch.matmul(W_enc1, N)  # shape: [l]
    exp_WN = torch.exp(WN)        # shape: [l]
    bm_vec = exp_WN / (1 + (exp_WN - 1) * e1.squeeze())
    bm = bm_vec.sum()

    # --- Scaling ---
    scale_FM = (1.5 * n) / (math.sqrt(2) * eps[I])
    if app_sen_noise and use_bm:
        scale_FM *= bm
    if const_val>0:
        e1b=e1*const_val
        cj=torch.sum(e1b)
        scale_FM+= n*cj*(0.5*cj+2)

    # --- Shared terms ---
    f_ji1 = math.log(2)
    f_ji2 = 0.5 - y_pred.squeeze()
    f_ji3 = 0.5 * y_pred.squeeze() - 0.25

    # --- Vectorized computations for both users ---
    a1 = torch.matmul(e1, W_dec[0:l, :])    # shape: [n]
    b1 = a1 ** 2

    a2 = torch.matmul(e2, W_dec[l:2*l, :])  # shape: [n]
    b2 = a2 ** 2

    if app_FM_DP:
        noise = torch.distributions.Laplace(0, scale_FM).sample([n]).to(device)
        if app_sen_noise:
            f_ji2=f_ji2*bm
            f_ji3=f_ji3*bm
        sum1 = torch.sum(f_ji1 + (f_ji2 + noise) * a1 + (f_ji3 + noise) * b1)
        sum2 = torch.sum(f_ji1 + (f_ji2 + noise) * a2 + (f_ji3 + noise) * b2)
    elif dpsgd==1:
        f_ji2=f_ji2*bm
        f_ji3=f_ji3*bm
        sum1 = torch.sum(f_ji1 + f_ji2 * a1 + f_ji3 * b1)
        sum2 = torch.sum(f_ji1 + f_ji2 * a2 + f_ji3 * b2)
    else:
        sum1 = torch.sum(f_ji1 + f_ji2 * a1 + f_ji3 * b1)
        sum2 = torch.sum(f_ji1 + f_ji2 * a2 + f_ji3 * b2)
    
    return sum1 + sum2


with open('fitbit_dataset.json') as f:
    inp = np.array(json.load(f))

inp = np.reshape(inp, (tot_data_sz, n)).astype(np.float32)
for i in range(n):
    inp[:, i] /= np.max(inp[:, i])

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
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)


def findbm(model, x, sen_noise_sig, n):
    device = x.device
    W_dec = model.decoder.weight.T  # shape [2l, n]
    W_enc1 = model.encoder1.weight  # shape [l, n]
    e1 = torch.sigmoid(model.encoder1(x))  # shape [l]
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
print('bm: ',bm)

if dpsgd==1:
    scale_dpsgd=(n+2*n*C)/(2*eps[I])
    if use_bm==1:
        scale_dpsgd=(n*bm+2*n*C)/(2*eps[I])
    privacy_engine = PrivacyEngine(model, batch_size=1, sample_size=len(train_set), epochs=10, target_epsilon=eps[I], noise_multiplier=scale_dpsgd, clipping_fn='Abadi', clipping_mode='ghost', clipping_style='all-layer',
    max_grad_norm=C)
else:
    privacy_engine = PrivacyEngine(model, batch_size=1, sample_size=len(train_set), epochs=10, target_epsilon=None, noise_multiplier=0.0, clipping_fn='none', clipping_mode='none', origin_params=None,) #non-private as FM noise in injected into loss function
privacy_engine.attach(optimizer) #Laplace noise is added in fast-differential-privacy/fastDP/supported_layers_grad_samplers.py which uses noise_multiplier as the s.d.

start_time = time.time()

for epoch in range(1):
    model.train()
    train_loss = 0.0
    
    for x, x_hat in train_loader:
        #shift_decoder_weights(model, constant=0.75)
        x, x_hat = x.to(device), x_hat.to(device)
        optimizer.zero_grad()

        y_pred, e1, e2 = model(x) #forward pass invokes out = torch.sigmoid(self.decoder_gain * self.decoder(dec_input))
        W_dec = model.decoder.weight.T
        W_enc1 = model.encoder1.weight

        if use_custom_loss:
            loss = my_loss(W_dec, e1, e2, x_hat, y_pred, W_enc1)
            #print('loss: ', loss)
        else:
            #loss = F.binary_cross_entropy(y_pred, x_hat)
            loss = F.binary_cross_entropy_with_logits(y_pred, x_hat)

        loss.backward() #back prop
        
        '''
        with open(f"gradlog_full_I{I}.txt", "a") as f:
            f.write(f"\n[Epoch {epoch}]\n")
            for name, param in model.named_parameters():
                raw_grad = param.grad.norm().item() if param.grad is not None else None
                private_grad = param.private_grad.norm().item() if hasattr(param, "private_grad") else None
                noise_scale = getattr(param, "noise", None)
                f.write(f"{name:30s} | raw grad: {raw_grad:.2f} | clipped+noised: {private_grad:.2f} | noise scale: {noise_scale}\n")
        '''
        
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}")
    
end_time = time.time()
tot_time=end_time - start_time
print(f"Total simulation time: {tot_time:.2f} seconds")


# ----- Evaluation -----
model.eval()
with torch.no_grad():
    X_pred, _, _ = model(X_hat)  # forward pass
# Compute MSE and accuracy
mse = torch.mean((X_pred - X) ** 2).item()
acc = 1 - mse
print(f"Accuracy (1 - MSE): {acc:.6f}")

if dpsgd == 1:
    filename = f"dpsgdaccuracy_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif app_sen_noise==1:
    filename = f"FMaccuracy_noisyInp_{I}_{use_bm}_{sen_noise_sig}.txt"
elif app_sen_noise == 0 and use_custom_loss == 1 and app_FM_DP == 1:
    filename = f"FMaccuracy_noislessInp_{I}.txt"
elif privacy_mode=='nonprivate':
    filename = f"nonprivate_{I}.txt"
with open(filename, 'a') as f:
    f.write(str(acc) + " ")

if privacy_mode=='nonprivate':
    filename = f"nonPrivate_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode=='dpsgd':
    filename = f"dpsgd_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
elif privacy_mode=='fm':
    filename = f"fm_time_{I}_{use_custom_loss}_{app_sen_noise}_{sen_noise_sig}.txt"
with open(filename, 'a') as f:
    f.write(str(tot_time) + " ")
