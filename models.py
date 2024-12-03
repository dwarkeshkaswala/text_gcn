"""
Neural Network Models for Text GCN

This module implements the core model architectures:
- Base Model: Common functionality for all models
- MLP (Multi-Layer Perceptron): For baseline comparison
- GCN (Graph Convolutional Network): Main architecture

Features:
- TensorFlow 2.x implementation
- Sparse matrix support
- Dropout regularization
- Weight decay
- Model saving/loading
- Masked loss and accuracy computation
"""

import tensorflow as tf
from layers import GraphConvolution, Dense
from metrics import masked_softmax_cross_entropy, masked_accuracy

class BaseModel(tf.keras.Model):
    """Base class with common functionality for all models."""
    
    def __init__(self, name=None, logging=False, **kwargs):
        """Initialize base model with logging capability."""
        super(BaseModel, self).__init__(name=name)
        self.logging = logging

    def save(self):
        self.save_weights(f"tmp/{self.name}.ckpt")
        print(f"Model saved in file: tmp/{self.name}.ckpt")

    def load(self):
        self.load_weights(f"tmp/{self.name}.ckpt")
        print(f"Model restored from file: tmp/{self.name}.ckpt")

class MLP(BaseModel):
    def __init__(self, input_dim, output_dim, args, **kwargs):
        # Extract and remove custom arguments
        logging = kwargs.pop('logging', False)
        name = kwargs.pop('name', None)
        super(MLP, self).__init__(name=name, logging=logging)
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = args.dropout
        self.build_model()

    def build_model(self):
        self.dense1 = Dense(input_dim=self.input_dim,
                            output_dim=self.args.hidden1,
                            activation=tf.nn.relu,
                            dropout_rate=self.dropout_rate,
                            sparse_inputs=True,
                            logging=self.logging)

        self.dense2 = Dense(input_dim=self.args.hidden1,
                            output_dim=self.output_dim,
                            activation=lambda x: x,
                            dropout_rate=self.dropout_rate,
                            logging=self.logging)

    def call(self, inputs, training=False):
        x = self.dense1(inputs, training=training)
        x = self.dense2(x, training=training)
        return x

    def compute_loss(self, inputs, labels, mask):
        logits = self.call(inputs, training=True)
        loss = masked_softmax_cross_entropy(logits, labels, mask)
        # Weight decay loss
        for var in self.trainable_variables:
            loss += self.args.weight_decay * tf.nn.l2_loss(var)
        return loss

    def compute_accuracy(self, inputs, labels, mask):
        logits = self.call(inputs, training=False)
        accuracy = masked_accuracy(logits, labels, mask)
        return accuracy

class GCN(BaseModel):
    def __init__(self, input_dim, output_dim, support, args, **kwargs):
        # Extract and remove custom arguments
        logging = kwargs.pop('logging', False)
        name = kwargs.pop('name', None)
        super(GCN, self).__init__(name=name, logging=logging)
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = support  # Save the support
        self.dropout_rate = args.dropout
        self.build_model()

    def build_model(self):
        self.gc1 = GraphConvolution(input_dim=self.input_dim,
                                    output_dim=self.args.hidden1,
                                    support=self.support[0],  # Pass the support
                                    activation=tf.nn.relu,
                                    dropout_rate=self.dropout_rate,
                                    sparse_inputs=True,
                                    featureless=self.args.featureless,
                                    logging=self.logging)

        self.gc2 = GraphConvolution(input_dim=self.args.hidden1,
                                    output_dim=self.output_dim,
                                    support=self.support[0],  # Pass the support
                                    activation=lambda x: x,
                                    dropout_rate=self.dropout_rate,
                                    logging=self.logging)

    def call(self, inputs, training=False):
        x = inputs  # Only node features
        x = self.gc1(x, training=training)
        x = self.gc2(x, training=training)
        return x

    def compute_loss(self, inputs, labels, mask):
        logits = self.call(inputs, training=True)
        loss = masked_softmax_cross_entropy(logits, labels, mask)
        # Weight decay loss
        for var in self.trainable_variables:
            loss += self.args.weight_decay * tf.nn.l2_loss(var)
        return loss

    def compute_accuracy(self, inputs, labels, mask):
        logits = self.call(inputs, training=False)
        accuracy = masked_accuracy(logits, labels, mask)
        return accuracy
