"""
Neural Network Layer Implementations for Text GCN

This module implements specialized layers for Graph Convolutional Networks (GCN):

Core Components:
1. Base Layer Class
   - Provides logging functionality
   - Handles variable management
   - Implements common layer operations

2. Dense Layer
   - Supports sparse inputs
   - Configurable dropout
   - Flexible activation functions
   - Weight regularization

3. Graph Convolution Layer
   - Implements graph convolution operations
   - Handles adjacency matrix operations
   - Supports feature aggregation
   - Configurable for different graph structures

Technical Features:
- Sparse tensor optimization
- CPU/GPU compatibility
- Memory-efficient operations
- Support for large-scale graphs

Usage Example:
    gc_layer = GraphConvolution(
        input_dim=input_dim,
        output_dim=FLAGS.hidden1,
        support=support[0],
        activation=tf.nn.relu,
        dropout_rate=FLAGS.dropout,
        sparse_inputs=True
    )
"""

from inits import *
import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Assigns unique layer IDs for each layer type.
    
    This function ensures each layer instance gets a unique identifier,
    which is useful for variable naming and debugging.
    
    Args:
        layer_name (str): Base name for the layer
        
    Returns:
        int: Unique ID for this layer type
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Performs dropout operation on sparse tensors.
    
    Efficiently implements dropout for sparse tensors by only dropping
    non-zero elements.
    
    Args:
        x: Sparse tensor input
        keep_prob: Probability of keeping each element
        noise_shape: Shape of the dropout mask
        
    Returns:
        Sparse tensor with dropout applied
    """
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Performs matrix multiplication handling both sparse and dense inputs.
    
    Wrapper function that chooses the appropriate multiplication operation
    based on input sparsity.
    
    Args:
        x: First input tensor (sparse or dense)
        y: Second input tensor
        sparse: Boolean indicating if x is sparse
        
    Returns:
        Result of matrix multiplication
    """
    if sparse:
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    
    This class provides common functionality for all layers:
    - Variable management
    - Logging capabilities
    - Basic layer operations
    
    Properties:
        name: String, defines the variable scope of the layer
        logging: Boolean, switches Tensorflow histogram logging on/off
        
    Methods:
        _call(inputs): Defines computation graph of layer
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim,
                 support,
                 activation=tf.nn.relu,
                 dropout_rate=0.,
                 sparse_inputs=False,
                 featureless=False,
                 **kwargs):
        # Extract custom arguments and remove them from kwargs
        self.logging = kwargs.pop('logging', False)
        self.featureless = featureless
        self.sparse_inputs = sparse_inputs
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.support = support  # Save the support (adjacency matrix)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Call the base Layer's __init__ with remaining kwargs
        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create the variables here
        if self.featureless:
            # If featureless, x is not used, and pre_sup is kernel directly
            num_nodes = self.support.shape[1]
            self.kernel = self.add_weight(shape=(num_nodes, self.output_dim),
                                          initializer='glorot_uniform',
                                          name='kernel')
        else:
            self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
                                          initializer='glorot_uniform',
                                          name='kernel')
        # Initialize bias if needed
        self.bias = self.add_weight(shape=(self.output_dim,),
                                    initializer='zeros',
                                    name='bias')
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, training=False):
        if not self.featureless:
            x = inputs  # Node features

            if self.sparse_inputs:
                x = tf.sparse.reorder(x)
                x = tf.sparse.to_dense(x)

            if training and self.dropout_rate > 0.0:
                x = tf.nn.dropout(x, rate=self.dropout_rate)

            pre_sup = tf.matmul(x, self.kernel)
        else:
            # If featureless, x is not used, and pre_sup is kernel directly
            pre_sup = self.kernel

        # Multiply with support (adjacency matrix)
        with tf.device('/CPU:0'):
            output = tf.sparse.sparse_dense_matmul(self.support, pre_sup)

        if self.bias is not None:
            output += self.bias

        return self.activation(output)