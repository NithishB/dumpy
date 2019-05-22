import numpy as np
import tensorflow as tf

def activate(x, function="relu", grad=False):
    if grad:
        if function == "relu":
            return 1.*tf.cast((x>0),tf.float32)
        elif function =="sigmoid":
            return x*(x-1)
        elif function == "tanh":
            return 1. - np.tanh(x)**2
        elif function == "elu":
            alpha=0.01
            return 1.*tf.cast((x>0),tf.float32) + alpha*np.exp(x)*tf.cast((x<=0),tf.float32)
        else:
            return x
    else:
        if function == "relu":
            return x*tf.cast((x>0),tf.float32)
        elif function == "sigmoid":
            return 1./(1+np.exp(-x))
        elif function == "tanh":
            return np.tanh(x)
        elif function =="elu":
            return x*tf.cast((x>0),tf.float32) + alpha*(np.exp(x)-1)*tf.cast((x<=0),tf.float32)
        else:
            return np.ones_like(x)