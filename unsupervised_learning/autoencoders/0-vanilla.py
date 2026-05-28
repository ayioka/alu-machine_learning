#!/usr/bin/env python3
"""Vanilla Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder"""

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes,
                               activation='relu')(x)

    latent = keras.layers.Dense(latent_dims,
                                activation='relu')(x)

    encoder = keras.Model(inputs, latent)

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes,
                               activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs,
                          outputs)

    auto_inputs = inputs
    auto_outputs = decoder(
        encoder(auto_inputs)
    )

    auto = keras.Model(auto_inputs,
                       auto_outputs)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
