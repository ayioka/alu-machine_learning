#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras

K = keras.backend


def sample(args):
    """Sampling"""

    mu, log_var = args

    epsilon = K.random_normal(
        shape=K.shape(mu)
    )

    return mu + K.exp(
        log_var / 2
    ) * epsilon


def autoencoder(input_dims,
                hidden_layers,
                latent_dims):
    """Creates VAE"""

    inputs = keras.Input(
        shape=(input_dims,)
    )

    x = inputs

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
        sample
    )([mu, log_var])

    encoder = keras.Model(
        inputs,
        [latent, mu, log_var]
    )

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

    reconstruction_loss = \
        keras.losses.binary_crossentropy(
            inputs,
            reconstructed
        )

    reconstruction_loss *= input_dims

    kl_loss = 1 + log_var
    kl_loss -= K.square(mu)
    kl_loss -= K.exp(log_var)

    kl_loss = K.sum(
        kl_loss,
        axis=-1
    )

    kl_loss *= -0.5

    total_loss = K.mean(
        reconstruction_loss +
        kl_loss
    )

    auto.add_loss(
        total_loss
    )

    auto.compile(
        optimizer='adam'
    )

    return encoder, decoder, auto
