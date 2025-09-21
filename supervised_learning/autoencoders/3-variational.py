#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: the encoder model (outputs latent, mean, log_var)
        decoder: the decoder model
        auto: the full autoencoder model
    """

    # ----- Encoder -----
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims, activation=None, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None, name="z_log_var")(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name="z")([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # ----- Decoder -----
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # ----- Autoencoder -----
    z, mu, log_var = encoder(inputs)
    reconstructed = decoder(z)
    auto = keras.Model(inputs, reconstructed, name="vae")

    # ----- Custom loss -----
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)

    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto

