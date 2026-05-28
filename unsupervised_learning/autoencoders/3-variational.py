#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras

K = keras.backend


def sampling(args):
    """sample latent vector"""

    mu, log_var = args

    epsilon = K.random_normal(
        shape=K.shape(mu)
    )

    return mu + K.exp(
        log_var / 2
    ) * epsilon


def autoencoder(
        input_dims,
        hidden_layers,
        latent_dims):
    """creates VAE"""

    inputs = keras.Input(
        shape=(input_dims,)
    )

    x = inputs

    # encoder hidden layers

    for nodes in hidden_layers:

        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    mu = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    latent = keras.layers.Lambda(
        sampling
    )([mu, log_var])

    encoder = keras.Model(
        inputs,
        [latent, mu, log_var]
    )

    # decoder

    latent_inputs = keras.Input(
        shape=(latent_dims,)
    )

    x = latent_inputs

    for nodes in hidden_layers[::-1]:

        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        latent_inputs,
        outputs
    )

    reconstructed = decoder(
        latent
    )

    auto = keras.Model(
        inputs,
        reconstructed
    )

    reconstruction = \
        keras.losses.binary_crossentropy(
            inputs,
            reconstructed
        )

    reconstruction *= input_dims

    kl = 1 + log_var
    kl -= K.square(mu)
    kl -= K.exp(log_var)

    kl = K.sum(
        kl,
        axis=-1
    )

    kl *= -0.5

    auto.add_loss(
        K.mean(
            reconstruction + kl
        )
    )

    auto.compile(
        optimizer='adam'
    )

    return encoder, decoder, auto
