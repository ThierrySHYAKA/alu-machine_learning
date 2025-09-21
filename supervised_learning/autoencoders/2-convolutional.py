#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims (tuple): dimensions of the model input (H, W, C)
        filters (list): filters for each convolutional layer in encoder
        latent_dims (tuple): dimensions of the latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # ----- Encoder -----
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Latent representation
    latent = x
    encoder = keras.Model(inputs, latent, name="encoder")

    # ----- Decoder -----
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for i, f in enumerate(reversed(filters)):
        if i < len(filters) - 2:  # all but last two convs
            x = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                activation='relu',
                padding='same'
            )(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(filters) - 2:  # second to last conv
            x = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                activation='relu',
                padding='valid'
            )(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        else:  # last conv → output
            x = keras.layers.Conv2D(
                filters=input_dims[-1],
                kernel_size=(3, 3),
                activation='sigmoid',
                padding='same'
            )(x)

    decoder = keras.Model(latent_inputs, x, name="decoder")

    # ----- Autoencoder -----
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="conv_autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

