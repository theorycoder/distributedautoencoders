#use script $ for i in {1..14}; do python3 autoencoder4.py; done
#four encoders, 1 decoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import tensorflow as tf
import tensorflow_datasets as tfds
import collections
from keras.src import ops
from keras import backend as K
import numpy as np
import random
import math
import json
from accountant import *
from sanitizer import *
from pathlib import Path
#tf.compat.v1.disable_eager_execution()
EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

C = 4
BATCH_SIZE=1
n=14
l=7 #compression rate=l/n
tot_data_sz=456
num_user=4
dpsgd=0
if dpsgd:
    use_cl=1
    app_FM_DP = 0 
    app_sen_noise =  1 
    useb=0
    sen_noise_sig = 5 
else:     
    use_cl = 1 #int(input ("apply custom loss? "))
    app_FM_DP = 1 #int(input ("apply DP noise? "))
    app_sen_noise =  0 #int(input ("apply sensor noise? "))
    sen_noise_sig = 1 #sensor noise s.d.
    #if app_sen_noise==1:
    #    useb = int(input ("scale noise with b? "))
    useb=1
I = int(input ("privacy budget index "))
eps=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
scale_dpsgd=(2*n*np.log(2))/eps[I] #cross entropy has ln not base 10 log
#scale_dpsgd=(2*n*C)/eps[I] 
#print('np.log(2) ',np.log(2))

print('dpsgd: ',dpsgd)
print('use_cl: ',use_cl)
print('app_FM_DP: ',app_FM_DP)
print('app_sen_noise: ',app_sen_noise)
print('useb: ',useb)

class Dense(tf.Module):
  def __init__(self, in_features, out_features, activation, name=None):
    super().__init__(name=name)
    self.activation = activation
    self.w = tf.Variable(tf.initializers.GlorotUniform()([in_features, out_features]), name='weights')
    self.b = tf.Variable(tf.zeros([out_features]), name='biases')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return self.activation(y)


class autoencoder(tf.Module):
  def __init__(self, name):
    super().__init__(name=name)
    self.encoder1 = Dense(in_features=n, out_features=l, activation=tf.nn.sigmoid, name='enf_ji2')
    self.encoder2 = Dense(in_features=n, out_features=l, activation=tf.nn.sigmoid)
    self.encoder3 = Dense(in_features=n, out_features=l, activation=tf.nn.sigmoid)
    self.encoder4 = Dense(in_features=n, out_features=l, activation=tf.nn.sigmoid)
    self.decoder = Dense(in_features=num_user*l, out_features=n, activation=tf.nn.sigmoid)
    
	
  def __call__(self, x):
    encoded1 = self.encoder1(x)
    encoded2 = self.encoder2(x)
    encoded3 = self.encoder3(x)
    encoded4 = self.encoder4(x)
    decoded = self.decoder(tf.concat([encoded1, encoded2, encoded3, encoded4], 1))
    
    return decoded
    

    
def my_loss(W_dec, enc1_out, enc2_out, enc3_out, enc4_out, y_true, y_pred): #custom loss has to be a differential function
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	sum1=tf.constant(0.)
	sum2=tf.constant(0.)
	sum3=tf.constant(0.)
	sum4=tf.constant(0.)
		
	W_enc1=model.encoder1.w
	#print('', np.shape(W_enc1))
	#print('', np.shape(enc1_out[0,0]))
	#print('enc1_out: ', enc1_out)
	#print('', enc1_out[0,0])
	
	N=np.random.normal(0,sen_noise_sig,n)
	WN=np.zeros(len(enc1_out))
	for i in range(len(enc1_out)):
	    WN[i]=np.tensordot(W_enc1[0:int(n),i], N, axes=1)
	#print('WN: ',WN)
	b1=0
	for i in range(len(enc1_out)):
	    b1i=math.exp(WN[i])/(1+(math.exp(WN[i])-1)*enc1_out[0,i])
	    b1=b1+b1i #for user 1
	#print('b1: ',b1)
	
	scale_FM=(1.5*n)/(math.sqrt(2)*eps[I])
	#print('scale_FM: ',scale_FM)
	
	for i in range(n): 
		f_ji1=math.log(2)
		f_ji2=0.5-y_pred[:,i]
		f_ji3=0.5*y_pred[:,i]-0.25 #: in y_pred is over all realizations of a user data
		W_dec=W_dec +0.5 #improves numerical stability, otherwise weights are too small	

		if app_sen_noise==1 and useb==1:
		    scale_FM=scale_FM*b1
		    
		#print('shape.W_dec: ', np.shape(W_dec[0:7,0]))
		#print('shape.enc1_out[0]: ', np.shape(enc1_out[0]))
		a=np.tensordot(enc1_out, W_dec[0:l,i], axes=1) 
		b=tf.math.square(np.tensordot(enc1_out, W_dec[0:l,i], axes=1))
		noise3=np.random.laplace(0,scale_FM,1)
		A=f_ji1 + (f_ji2+noise3)*a + (f_ji3+noise3)*b
		B=f_ji1 + f_ji2*a + f_ji3*b
		
		if app_FM_DP:
			sum1 = sum1 + A
		else:
			sum1 = sum1 + B #summation over i for a given user j
		#print('sum1: ',sum1)	
		
		a=np.tensordot(enc2_out, W_dec[l:2*l,i], axes=1) 
		b=tf.math.square(np.tensordot(enc2_out, W_dec[l:2*l,i], axes=1))
		A=f_ji1 + (f_ji2+noise3)*a + (f_ji3+noise3)*b
		B=f_ji1 + f_ji2*a + f_ji3*b
		if app_FM_DP:
			sum2 = sum2 + A #for another j
		else:
			sum2 = sum2 + B 
		
		a=np.tensordot(enc3_out, W_dec[2*l:3*l,i], axes=1) 
		b=tf.math.square(np.tensordot(enc3_out, W_dec[2*l:3*l,i], axes=1))
		A=f_ji1 + (f_ji2+noise3)*a + (f_ji3+noise3)*b
		B=f_ji1 + f_ji2*a + f_ji3*b
		if app_FM_DP:
			sum3 = sum3 + A #for another j
		else:
			sum3 = sum3 + B 
		
		a=np.tensordot(enc4_out, W_dec[3*l:4*l,i], axes=1) 
		b=tf.math.square(np.tensordot(enc4_out, W_dec[3*l:4*l,i], axes=1))
		A=f_ji1 + (f_ji2+noise3)*a + (f_ji3+noise3)*b
		B=f_ji1 + f_ji2*a + f_ji3*b
		if app_FM_DP:
			sum4 = sum4 + A #for another j
		else:
			sum4 = sum4 + B 
        
	sum1 = tf.cast(sum1, tf.float32) 
	sum2 = tf.cast(sum2, tf.float32)
	sum3 = tf.cast(sum3, tf.float32)
	sum4 = tf.cast(sum4, tf.float32)
	#print('val: ', val)
	
	return (sum1+sum2+sum3+sum4) #summation over j
	
		    
model = autoencoder(name='sequential_model')
loss_object = tf.losses.BinaryCrossentropy(from_logits=False)

def compute_loss(model, x, x_hat):
	W_dec=model.decoder.w
	print('W_dec shape: ', W_dec.shape)
	enc1_out = model.encoder1(x).numpy()
	enc2_out = model.encoder2(x).numpy()
	enc3_out = model.encoder3(x).numpy()
	enc4_out = model.encoder4(x).numpy()
	#print('enc1_out shape: ', enc1_out.shape)
	
	x_pred = model(x)
	if use_cl==1:
		loss = my_loss(W_dec, enc1_out, enc2_out, enc3_out, enc4_out, y_true=x_hat, y_pred=x_pred)
	else:
		loss = loss_object(y_true=x_hat, y_pred=x_pred)
	
	return loss, x_pred


def get_grad(model, x, y):
    with tf.GradientTape() as tape:
        loss, out = compute_loss(model, x, y)
        #print('loss :', loss)
        gradients = tape.gradient(loss, model.trainable_variables) #immutable
        #print("gradients[0]: ", gradients[0])
        
    return loss, gradients, out

optimizer = tf.optimizers.Adam()
verbose = "Epoch {:2d} Loss: {:.3f} TLoss:you must get rid of personal identifiers and even parts of them (like initials).  {:.3f} Acc: {:=7.2%} TAcc: {:=7.2%}"



#************************************************************* 

with open('fitbit_dataset.json') as f:
    inp = json.load(f)
#print('inp: ',inp)
print('inp shape: ',np.shape(inp))
inp = np.reshape(inp, (tot_data_sz,n))

#X=np.zeros(shape=(tot_data_sz,n)) 
X=inp
for i in range(n):
    maxcol= np.max(X[:, i])
    X[:, i]=X[:, i]/maxcol
#noise=np.random.normal(0,1,1)
noise1 = np.random.laplace(0,1,tot_data_sz*n)
noise1 = np.reshape(noise1, (tot_data_sz,n))

#sensor noise
noise2 = np.random.normal(0,sen_noise_sig,tot_data_sz*n) #noise scale is the s.d.
noise2 = np.reshape(noise2, (tot_data_sz,n))

X_pred = np.zeros(shape=(tot_data_sz,n))

for i in range(0, len(X[:,0])):
	for j in range(0, len(X[0,:])):
		#print('i: ',i)
		#print('j: ',j)
		if random.randint(0,100)>=90:
			X[i,j]=1

if app_FM_DP==1 and use_cl==0:
	X_hat=X+noise1 #DP noise
elif app_sen_noise==1:
	X_hat=X+noise2 #sensor noise
else:
    X_hat=X
#print('X shape: ', X.shape)
print('X: ', X)

X=X.astype(np.float32)
X_hat=X_hat.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((X,X_hat))
#print('dataset.element_spec: ', dataset.element_spec)

train = dataset.take(int(tot_data_sz*0.8)).shuffle(int(tot_data_sz*0.8)).batch(1)
test = dataset.skip(int(tot_data_sz*0.8)).take(int(0.2*tot_data_sz)).shuffle(int(0.2*tot_data_sz)).batch(1)
#print('train: ', train)
#print('test: ', test)

delta = 1e-7
max_eps = 64.0
max_delta = 1e-3
target_eps = [64.0]
target_delta = [1e-5] #unused
    
# Create accountant, sanitizer and metrics
accountant = AmortizedAccountant(n)
sanitizer = AmortizedGaussianSanitizer(accountant, [C / BATCH_SIZE, True])
spent_eps_delta = EpsDelta(0, 0)
should_terminate = False

for epoch in range(10):
	train_loss = tf.constant(0.)
	train_acc = tf.constant(0.)
	test_loss = tf.constant(0.)
	test_acc = tf.constant(0.)

	for n_train, (x, y) in enumerate(train, 1):
		#print('x: ', x)
		#print('y: ', y)		
		loss_value, grads, out = get_grad(model, x, y)
		#grads=np.array(grads)
		#print('grads shape: ', grads.shape)
		
		if dpsgd:
		    sanitized_grads = []
		    eps_delta = EpsDelta(eps[I], delta)
		    for px_grad in grads:
		        sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, scale_dpsgd)
		        sanitized_grads.append(sanitized_grad)
		    spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]
		    optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))
		    if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
		        should_terminate = True
		else:
		    optimizer.apply_gradients(zip(grads, model.trainable_variables))
		train_loss += loss_value
		train_acc += tf.metrics.sparse_categorical_accuracy(y, out)[0]
		
	for n_test, (x, y) in enumerate(test, 1):
		loss_value, _, out = get_grad(model, x, y)
		test_loss += loss_value
		test_acc += tf.metrics.sparse_categorical_accuracy(y, out)[0]
		
	n_train=tf.cast(n_train, tf.float32)
	n_test=tf.cast(n_test, tf.float32)	
		

X_pred=model(X_hat)	   		
X_pred = np.asarray(X_pred, dtype="object")
#print('X_pred: ',X_pred)
#mse = ((X_pred - X_hat)**2).mean(axis=None)
mse = ((X_pred - X)**2).mean(axis=None)
acc=1-mse
print('accuracy: ', acc)		

if dpsgd==1:
    filename="dpsgdaccuracy"+"_"+str(I)+"_"+str(use_cl)+"_"+str(sen_noise_sig)+".txt"
elif app_sen_noise==1:
    filename="FMaccuracy_noisyInp"+"_"+str(I)+"_"+str(useb)+"_"+str(sen_noise_sig)+".txt" #noisyInpB is when noise2 scale sigma=1 instead of 5
elif app_sen_noise==0 and use_cl == 1 and app_FM_DP == 1:
    filename="FMaccuracy_noislessInp"+"_"+str(I)+".txt"
elif app_sen_noise==0 and use_cl == 0 and app_FM_DP == 0:
    filename="nonprivate"+"_"+str(I)+".txt"
with open(filename, 'a') as f:
    f.write(str(acc)+ " ")
#a=np.loadtxt("my_file.txt") 

