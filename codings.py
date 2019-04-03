import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

def fc_uci_decoder(z,  obs_dim,  activation=tf.nn.sigmoid): #only output means since the model is N(m,sigmaI) or bernouli(m)
    x = layers.fully_connected(z, 50, scope='fc-01')
    x = layers.fully_connected(x, 100, scope='fc-02')
    x = layers.fully_connected(x, obs_dim, activation_fn=tf.nn.sigmoid,
                               scope='fc-final')

    return x, None

def fc_uci_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 100, scope='fc-01')
    e = layers.fully_connected(e, 50, scope='fc-02')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                               scope='fc-final')

    return e

def PNP_fc_uci_encoder(x, K, activation=None):
    e = layers.fully_connected(x, 100, scope='fc-01')
    e = layers.fully_connected(e, 50, scope='fc-02')
    e = layers.fully_connected(e, K, scope='fc-final')

    return e


