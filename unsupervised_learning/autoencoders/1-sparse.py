#!/usr/bin/env python3
"""Sparse Autoencoder implementation with L1 regularization"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder with L1 regularization"""
    # Encoder with L1 regularization on the latent layer
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    
    # Apply L1 regularization on the latent layer
    encoder_output = keras.layers.Dense(
        latent_dims, 
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.models.Model(encoder_input, encoder_output)
    
    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_input, decoder_output)
    
    # Autoencoder
    auto_input = keras.layers.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(auto_input, decoded)
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
