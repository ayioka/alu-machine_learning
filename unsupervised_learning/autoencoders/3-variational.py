#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
K = keras.backend


def sampling(args):
    """Reparameterization trick"""

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
    """Creates variational autoencoder"""

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
        sampling
    )([mu, log_var])

    encoder = keras.Model(
        inputs,
        [latent, mu, log_var]
    )

    latent_inputs = keras.Input(
        shape=(latent_dims,)
    )

    x = latent_inputs

    for nodes in reversed(
            hidden_layers):

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

    decoded = decoder(latent)

    auto = keras.Model(
        inputs,
        decoded
    )

    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs,
        decoded
    )

    reconstruction_loss *= input_dims

    kl_loss = -0.5 * K.sum(
        1 + log_var
        - K.square(mu)
        - K.exp(log_var),
        axis=-1
    )

    vae_loss = K.mean(
        reconstruction_loss +
        kl_loss
    )

    auto.add_loss(
        vae_loss
    )

    auto.compile(
        optimizer='adam'
    )

    return encoder, decoder, auto
