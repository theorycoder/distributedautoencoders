import math
import numpy
import tensorflow as tf

def BatchClipByL2norm(t, upper_bound, name=None):
    assert upper_bound > 0
    saved_shape = tf.shape(t)
    #batch_size = tf.slice(saved_shape, [0], [1])
    batch_size = tf.constant(1.0)
    shape = tf.concat(0, [[tf.constant(128)], [-1]])
    t2 = tf.reshape(t, shape)
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                            tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.math.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
    return clipped_t

def AddGaussianNoise(t, sigma, name=None):
  noisy_t = t + tf.random.normal(tf.shape(t), stddev=sigma)
  return noisy_t
  
def AddLaplaceNoise(t, sigma, name=None):
  noisy_t = t + numpy.random.laplace(0, sigma, tf.shape(t))
  return noisy_t

def GetTensorOpName(x):
  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]
