"""
Utility Functions for Text GCN

This module provides essential utility functions for:
- Data loading and preprocessing
- Graph construction and manipulation
- Matrix operations
- Text cleaning and normalization
- Evaluation metrics

Key Components:
- Sparse matrix handling
- Graph adjacency matrix processing
- Feature normalization
- Mask generation for semi-supervised learning
- Word vector operations

Implementation Notes:
- Optimized for large sparse matrices
- Supports both CPU and GPU operations
- Handles multiple dataset formats
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import re

def parse_index_file(filename):
    """Parse index file containing document indices.
    
    Args:
        filename (str): Path to index file
        
    Returns:
        list: List of integer indices
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask array for semi-supervised learning.
    
    Args:
        idx: Indices to mark as True
        l: Total length of mask
        
    Returns:
        np.array: Boolean mask array
    """
    mask = np.zeros(l)
    print("Creating mask...")
    print("idx:", idx)
    print("max idx:", np.max(idx))
    print("mask length:", l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_corpus(dataset_str):
    """Load and preprocess the text corpus and associated data.
    
    Loads various components:
    - Feature matrices (x, tx, allx)
    - Labels (y, ty, ally)
    - Adjacency matrix (adj)
    - Training indices
    
    Args:
        dataset_str (str): Name of the dataset to load
        
    Returns:
        tuple: Contains all necessary data components for the GCN
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print("Shapes of loaded data:")
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("tx.shape:", tx.shape)
    print("ty.shape:", ty.shape)
    print("allx.shape:", allx.shape)
    print("ally.shape:", ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print("labels.shape:", labels.shape)

    train_idx_orig = parse_index_file("data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = 0  # Adjusted validation size
    test_size = ty.shape[0]  # Use the number of test labels

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    # Debug prints to check indices
    print("idx_train:", idx_train)
    print("max idx_train:", max(idx_train))
    print("idx_val:", idx_val)
    print("max idx_val:", max(idx_val) if val_size > 0 else "No validation indices")
    print("idx_test:", idx_test)
    print("max idx_test:", max(idx_test))
    print("labels.shape[0]:", labels.shape[0])

    # Ensure indices are within bounds
    assert max(idx_train) < labels.shape[0], "idx_train out of bounds"
    if val_size > 0:
        assert max(idx_val) < labels.shape[0], "idx_val out of bounds"
    assert max(idx_test) < labels.shape[0], "idx_test out of bounds"

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0]) if val_size > 0 else np.zeros(labels.shape[0], dtype=bool)
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    if val_size > 0:
        y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # Symmetrize adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.
    
    Args:
        sparse_mx: Sparse matrix or list of sparse matrices
        
    Returns:
        tuple or list of tuples: (coords, values, shape)
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            if(len(row) > 2):
                vocab.append(row[0])
                vector = [float(x) for x in row[1:]]
                embd.append(vector)
                word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    return vocab, embd, word_vector_map

def clean_str(string):
    """Clean and normalize text string.
    
    Performs operations like:
    - Lowercase conversion
    - Punctuation standardization
    - Contraction handling
    - Whitespace normalization
    
    Args:
        string (str): Input text string
        
    Returns:
        str: Cleaned and normalized string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
