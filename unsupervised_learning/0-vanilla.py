#!/usr/bin/env python3
"""
Module for creating a vanilla autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder model
    
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer in the encoder
        latent_dims: integer containing the dimensions of the latent space representation
    
    Returns:
        encoder: the encoder model
        decoder: the decoder model  
        auto: the full autoencoder model
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dims,))
    
    # Build encoder
    encoded = input_layer
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    
    # Latent layer
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    
    # Create encoder model
    encoder = keras.Model(input_layer, encoded)
    
    # Build decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    
    # Reverse the hidden layers for decoder
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    
    # Output layer with sigmoid activation
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    
    # Create decoder model
    decoder = keras.Model(decoder_input, decoded)
    
    # Create full autoencoder
    auto_encoded = encoder(input_layer)
    auto_decoded = decoder(auto_encoded)
    auto = keras.Model(input_layer, auto_decoded)
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
