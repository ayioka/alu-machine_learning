#!/usr/bin/env python3
"""Variational Autoencoder implementation"""

import tensorflow.keras as keras
import tensorflow as tf


def sampling(args):
    """Sampling function for VAE"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""
    
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    
    # Sampling layer
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    
    encoder = keras.models.Model(
        encoder_input, [z, z_mean, z_log_var], name='encoder'
    )
    
    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, decoder_output, name='decoder')
    
    # Autoencoder
    auto_input = keras.layers.Input(shape=(input_dims,))
    z, z_mean_vae, z_log_var_vae = encoder(auto_input)
    decoded = decoder(z)
    auto = keras.models.Model(auto_input, decoded, name='vae')
    
    # Custom loss
    reconstruction_loss = keras.losses.binary_crossentropy(auto_input, decoded)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var_vae - tf.square(z_mean_vae) - tf.exp(z_log_var_vae)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')
    
    return encoder, decoder, autoo
