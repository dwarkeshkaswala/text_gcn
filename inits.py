"""
Weight Initialization Functions

This module provides various initialization strategies for neural network weights:
- Uniform initialization within a range
- Glorot/Xavier initialization
- Zero initialization
- Ones initialization

These functions are used by the GCN and MLP layers to initialize their weights
in a way that promotes good training dynamics.
"""

import numpy as np
import tensorflow as tf


def uniform(shape, scale=0.05, name=None):
    """Uniform initialization within [-scale, scale]."""
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    shape = [s.value if hasattr(s, 'value') else s for s in shape]
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)