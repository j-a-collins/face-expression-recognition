"""
The main module for training a Siamese neural network for 
face recognition via Keras

Author: J-A-Collins
"""

# Imports
import utils
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

faces_dir = 'faces/'

def create_siamese_nn(input_shape, shared_network):
    """
    Create a Siamese Neural Network model.
    
    Parameters
    ----------
        input_shape:
            The shape of the input images

        shared_network:
            The shared network used in the Siamese Neural Network

    Returns
    -------
        A compiled Siamese Neural Network model

    """
    input_top = Input(shape=input_shape)
    input_bottom = Input(shape=input_shape)
    output_top = shared_network(input_top)
    output_bottom = shared_network(input_bottom)
    dist = Lambda(utils.euclidean_distance, output_shape=(1,))([output_top, output_bottom])
    model = Model(inputs=[input_top, input_bottom], outputs=dist)
    
    return model

def train_model(model, X_train, Y_train, batch_size=128, epochs=10):
    """
    Train the Siamese Neural Network model.
    
    Parameters
    ----------
        model:
            The Siamese Neural Network model

        X_train:
            Training images

        Y_train:
            Training labels

        batch_size:
            Batch size for training (default: 128)

        epochs:
            Number of epochs for training (default: 10)

    """
    num_classes = len(np.unique(Y_train))
    training_pairs, training_labels = utils.create_pairs(X_train, Y_train, num_classes=num_classes)
    model.compile(loss=utils.contrastive_loss, optimizer='adam', metrics=[utils.accuracy])
    model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
              batch_size=batch_size,
              epochs=epochs)

def save_model(model, file_name):
    """
    Save the trained model to a file.
    
    Parameters
    ----------
        model:
            The trained Siamese Neural Network model

        file_name:
            The name of the file to save the model

    """
    model.save(file_name)

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = utils.get_data(faces_dir)

# Create Siamese Neural Network
input_shape = X_train.shape[1:]
shared_network = utils.create_shared_network(input_shape)
model = create_siamese_nn(input_shape, shared_network)

# Train the model
train_model(model, X_train, Y_train)

# Save the model
save_model(model, 'nn.h5')
