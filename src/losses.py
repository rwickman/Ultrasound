import tensorflow as tf

def L1(p, y):
    return tf.reduce_mean(tf.abs(p - y))

def L2(p, y):
    return tf.sqrt(tf.reduce_mean(tf.abs(p - y) ** 2))

def Linf(p, y):
    return tf.reduce_max(tf.abs(p - y))

def Bias(p, y):
    return tf.reduce_mean(p - y)
