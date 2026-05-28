#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims,
                filters,
                latent_dims):
    """Creates convolutional autoencoder"""

    inputs = keras.Input(
        shape=input_dims
    )

    x = inputs

    for f in filters:

        x = keras.layers.Conv2D(
            f,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(
        inputs,
        x
    )

    latent_inputs = keras.Input(
        shape=latent_dims
    )

    x = latent_inputs

    rev_filters = filters[::-1]

    for f in rev_filters[:-1]:

        x = keras.layers.Conv2D(
            f,
            (3,3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.UpSampling2D(
            (2,2)
        )(x)

    x = keras.layers.Conv2D(
        rev_filters[-1],
        (3,3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        (2,2)
    )(x)

    outputs = keras.layers.Conv2D(
        input_dims[-1],
        (3,3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        latent_inputs,
        outputs
    )

    auto_outputs = decoder(
        encoder(inputs)
    )

    auto = keras.Model(
        inputs,
        auto_outputs
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
