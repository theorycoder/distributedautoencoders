#use script $ for i in {1..14}; do python3 autoencoder2_pytorch.py; done
#two encoders, 1 decoder
#This program uses the FastDP library available at https://github.com/thecml/dpsgd-optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
import json
from torch.utils.data import DataLoader, TensorDataset
from fastDP import PrivacyEngine

C = 4
n = 14
l = 7
tot_data_sz = 456
num_user = 2
dpsgd=0
if dpsgd:
    use_custom_loss=0
    app_FM_DP = 0 
    app_sen_noise=1 
    use_bm=1 #always 1
    sen_noise_sig = 5 
else:     
    use_custom_loss = 1 #int(input ("apply custom loss? "))
    app_FM_DP = 1 #int(input ("apply DP noise? "))
    app_sen_noise =  1 #int(input ("apply sensor noise? "))
    sen_noise_sig = 1 #sensor noise s.d.
    #if app_sen_noise==1:
    #    use_bm = int(input ("scale noise with b? "))
    use_bm=1
try:
    #I = int(input("Privacy budget index (0-6): ") or "0")  # default to 3
    I=6
except ValueError:
    print("Invalid input. Using default I=0")
    I = 0
eps = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


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
    sum1 = torch.tensor(0.0, device=device)
    sum2 = torch.tensor(0.0, device=device)

    N = torch.normal(0, sen_noise_sig, size=(n,), device=device)
    #WN = torch.tensordot(W_enc1.T, N, dims=([0], [0]))  
    #bm = sum([math.exp(WN[i].item()) / (1 + (math.exp(WN[i].item()) - 1) * e1[0, i].item()) for i in range(len(WN))])
    
    WN = torch.zeros(len(e1), device=W_enc1.device)  # shape [7]
    W_enc1_T = W_enc1.T  # shape [n, l] → [14, 7]
    for i in range(len(e1)):  # len(e1) = 7
        WN[i] = torch.dot(W_enc1_T[:, i], N)    
    #print('WN: ',WN)
    
    bm = 0
    for i in range(len(e1)):
        b1i = math.exp(WN[i].item()) / (1 + (math.exp(WN[i].item()) - 1) * e1[0, i].item())
        bm = bm + b1i  # for user 1
    #print('bm', bm)
    
    scale_FM = (1.5 * n) / (math.sqrt(2) * eps[I])
    if app_sen_noise and use_bm:
        scale_FM *= bm

    for i in range(n):
        f_ji1 = math.log(2)
        f_ji2 = 0.5 - y_pred[:, i]
        f_ji3 = 0.5 * y_pred[:, i] - 0.25
        W_dec = W_dec + 0.5 #improves numerical stability, otherwise weights are too small

        a = torch.tensordot(e1, W_dec[0:l, i], dims=1)
        b = torch.square(torch.tensordot(e1, W_dec[0:l, i], dims=1))
        noise3 = torch.distributions.Laplace(0, scale_FM).sample([1]).to(e1.device)
        
        if app_FM_DP:
            sum1 = sum1 + f_ji1 + (f_ji2 + noise3) * a + (f_ji3 + noise3) * b
        else:
            sum1 = sum1 + f_ji1 + f_ji2 * a + f_ji3 * b  # summation over i for a given user j
            
        a = torch.tensordot(e1, W_dec[l:n, i], dims=1)
        b = torch.square(torch.tensordot(e2, W_dec[l:n, i], dims=1))
        
        if app_FM_DP:
            sum2 = sum2 + f_ji1 + (f_ji2 + noise3) * a + (f_ji3 + noise3) * b  # for another j
        else:
            sum2 = sum2 + f_ji1 + f_ji2 * a + f_ji3 * b    
            
    return sum1 + sum2


with open('fitbit_dataset.json') as f:
    inp = np.array(json.load(f))

inp = np.reshape(inp, (tot_data_sz, n)).astype(np.float32)
for i in range(n):
    inp[:, i] /= np.max(inp[:, i])

X = inp.copy()
noise1 = np.random.laplace(0, 1, (tot_data_sz, n)).astype(np.float32)
noise2 = np.random.normal(0, sen_noise_sig, (tot_data_sz, n)).astype(np.float32)


for i in range(tot_data_sz):
    for j in range(n):
        if random.randint(0, 100) >= 90:
            X[i, j] = 1.0

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
device = torch.device("cpu")
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
    privacy_engine = PrivacyEngine(model, batch_size=256, sample_size=50000, epochs=3, target_epsilon=2, noise_multiplier=scale_dpsgd, clipping_fn='automatic', clipping_mode='MixOpt', clipping_style='all-layer')
else:
    privacy_engine = PrivacyEngine(model, batch_size=256, sample_size=50000, epochs=3, target_epsilon=None, noise_multiplier=0.0, clipping_fn='none', clipping_mode='none', origin_params=None,) #non-private as FM noise in injected into loss function
privacy_engine.attach(optimizer) #Laplace noise is added in fast-differential-privacy/fastDP/supported_layers_grad_samplers.py which uses noise_multiplier as the s.d.

for epoch in range(10):
    model.train()
    train_loss = 0.0

    for x, x_hat in train_loader:
        x, x_hat = x.to(device), x_hat.to(device)
        optimizer.zero_grad()

        y_pred, e1, e2 = model(x)
        W_dec = model.decoder.weight.T
        W_enc1 = model.encoder1.weight

        if use_custom_loss:
            loss = my_loss(W_dec, e1, e2, x_hat, y_pred, W_enc1)
        else:
            #loss = F.binary_cross_entropy(y_pred, x_hat)
            loss = F.binary_cross_entropy_with_logits(y_pred, x_hat)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}")

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
elif app_sen_noise == 0 and use_custom_loss == 0 and app_FM_DP == 0:
    filename = f"nonprivate_{I}.txt"

with open(filename, 'a') as f:
    f.write(str(acc) + " ")
