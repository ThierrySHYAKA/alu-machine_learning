#!/usr/bin/env python3
"""
Sparse Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation
        lambtha (float): L1 regularization parameter on the latent output

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full sparse autoencoder model
    """
    # ----- Encoder -----
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent layer with L1 activity regularizer
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # ----- Decoder -----
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ----- Sparse Autoencoder -----
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="sparse_autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

