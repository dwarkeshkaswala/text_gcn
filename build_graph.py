"""
Graph Construction Module for Text GCN

This script builds a heterogeneous graph from text documents:
- Constructs document-word edges based on TF-IDF
- Creates word-word edges based on PMI (Pointwise Mutual Information)
- Handles document splitting into train/test sets
- Generates and saves all necessary graph data structures
- Supports multiple datasets (20ng, R8, R52, ohsumed, mr)

The resulting graph contains:
- Document nodes
- Word nodes
- Weighted edges between documents and words
- Weighted edges between words based on co-occurrence

Usage:
    python build_graph.py <dataset>
where <dataset> is one of: 20ng, R8, R52, ohsumed, mr
"""

import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
import sys
import logging
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm  # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# Global variables (initialized in main)
dataset = None
vocab = None
word_embeddings_dim = 300  # Assuming word embeddings dimension is 300
word_vector_map = {}  # Placeholder for word vectors; adjust if using pre-trained embeddings
word_id_map = {}  # Will be assigned in build_vocabulary


def process_arguments():
    if len(sys.argv) != 2:
        sys.exit("Usage: python build_graph.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("Wrong dataset name. Available datasets: 20ng, R8, R52, ohsumed, mr")

    return dataset


def load_dataset(dataset):
    logger.info(f"Loading dataset: {dataset}")

    # Read document names and split into train/test
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []
    with open(f'data/{dataset}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.strip().split("\t")
            if 'test' in temp[1]:
                doc_test_list.append(line.strip())
            elif 'train' in temp[1]:
                doc_train_list.append(line.strip())

    # Read document contents
    doc_content_list = []
    with open(f'data/corpus/{dataset}.clean.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())

    return doc_name_list, doc_train_list, doc_test_list, doc_content_list


def shuffle_and_split_data(doc_name_list, doc_train_list, doc_test_list, doc_content_list, dataset):
    logger.info("Shuffling and splitting data into train and test sets.")

    train_ids = [doc_name_list.index(name) for name in doc_train_list]
    random.shuffle(train_ids)

    # Save train indices
    with open(f'data/{dataset}.train.index', 'w') as f:
        f.write('\n'.join(map(str, train_ids)))

    test_ids = [doc_name_list.index(name) for name in doc_test_list]
    random.shuffle(test_ids)

    # Save test indices
    with open(f'data/{dataset}.test.index', 'w') as f:
        f.write('\n'.join(map(str, test_ids)))

    ids = train_ids + test_ids
    shuffle_doc_name_list = [doc_name_list[i] for i in ids]
    shuffle_doc_words_list = [doc_content_list[i] for i in ids]

    # Save shuffled document names and contents
    with open(f'data/{dataset}_shuffle.txt', 'w') as f:
        f.write('\n'.join(shuffle_doc_name_list))

    with open(f'data/corpus/{dataset}_shuffle.txt', 'w') as f:
        f.write('\n'.join(shuffle_doc_words_list))

    return shuffle_doc_name_list, shuffle_doc_words_list, train_ids, test_ids


def build_vocabulary(shuffle_doc_words_list):
    logger.info("Building vocabulary.")

    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = list(word_set)
    vocab_size = len(vocab)
    word_id_map = {word: i for i, word in enumerate(vocab)}

    # Save vocabulary
    with open(f'data/corpus/{dataset}_vocab.txt', 'w') as f:
        f.write('\n'.join(vocab))

    return vocab, vocab_size, word_id_map, word_freq


def build_word_doc_list(shuffle_doc_words_list):
    logger.info("Building word-document list.")

    word_doc_list = defaultdict(list)
    for i, doc_words in enumerate(tqdm(shuffle_doc_words_list, desc="Processing documents")):
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word not in appeared:
                word_doc_list[word].append(i)
                appeared.add(word)

    word_doc_freq = {word: len(doc_ids) for word, doc_ids in word_doc_list.items()}

    return word_doc_list, word_doc_freq


def build_label_list(shuffle_doc_name_list):
    logger.info("Building label list.")

    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    # Save label list
    with open(f'data/corpus/{dataset}_labels.txt', 'w') as f:
        f.write('\n'.join(label_list))

    return label_list


def compute_document_vectors(shuffle_doc_words_list, indices):
    logger.info("Computing document vectors.")

    doc_vectors = {}
    for i in tqdm(indices, desc="Computing document vectors"):
        doc_words = shuffle_doc_words_list[i]
        doc_vec = np.zeros(word_embeddings_dim)
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec += np.array(word_vector)
        if doc_len > 0:
            doc_vec /= doc_len
        doc_vectors[i] = doc_vec

    return doc_vectors


def build_feature_matrices(doc_vectors, indices):
    logger.info("Building feature matrices.")

    data = []
    rows = []
    cols = []
    for idx in tqdm(indices, desc="Building feature matrices"):
        doc_vec = doc_vectors[idx]
        for j in range(word_embeddings_dim):
            data.append(doc_vec[j])
            rows.append(idx)
            cols.append(j)

    feature_matrix = sp.csr_matrix((data, (rows, cols)), shape=(max(indices) + 1, word_embeddings_dim))

    return feature_matrix


def build_labels(shuffle_doc_name_list, label_list, indices):
    logger.info("Building label matrices.")

    labels = []
    for i in tqdm(indices, desc="Building labels"):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0] * len(label_list)
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        labels.append(one_hot)

    labels = np.array(labels)

    return labels


def build_windows(shuffle_doc_words_list, window_size=20):
    logger.info("Building windows.")

    windows = []
    for doc_words in tqdm(shuffle_doc_words_list, desc="Processing documents for windows"):
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
    return windows


def compute_word_window_freq(windows):
    logger.info("Computing word window frequencies.")

    word_window_freq = {}
    for window in tqdm(windows, desc="Processing windows for word frequencies"):
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] = word_window_freq.get(word, 0) + 1
                appeared.add(word)
    return word_window_freq


def compute_word_pair_count(windows):
    logger.info("Computing word pair counts.")

    word_pair_count = {}
    for window in tqdm(windows, desc="Processing windows for word pairs"):
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_j = window[j]
                if word_i == word_j:
                    continue
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                pair_1 = (word_i_id, word_j_id)
                pair_2 = (word_j_id, word_i_id)
                for pair in [pair_1, pair_2]:
                    word_pair_count[pair] = word_pair_count.get(pair, 0) + 1

    return word_pair_count


def build_adjacency_matrix(train_size, vocab_size, test_size, word_pair_count, word_window_freq, word_doc_freq, windows, shuffle_doc_words_list):
    logger.info("Building adjacency matrix.")

    num_window = len(windows)
    row = []
    col = []
    weight = []

    # PMI between words
    for (i, j), count in tqdm(word_pair_count.items(), desc="Computing PMI for word pairs"):
        word_i = vocab[i]
        word_j = vocab[j]
        word_freq_i = word_window_freq[word_i]
        word_freq_j = word_window_freq[word_j]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # Doc-word frequencies
    logger.info("Computing document-word frequencies.")

    doc_word_freq = {}
    for doc_id in tqdm(range(len(shuffle_doc_words_list)), desc="Processing documents for doc-word frequencies"):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            key = (doc_id, word_id)
            doc_word_freq[key] = doc_word_freq.get(key, 0) + 1

    logger.info("Adding document-word edges to adjacency matrix.")

    for (doc_id, word_id), freq in tqdm(doc_word_freq.items(), desc="Adding edges to adjacency matrix"):
        if doc_id < train_size:
            row.append(doc_id)
        else:
            row.append(doc_id + vocab_size)
        col.append(train_size + word_id)
        idf = log(1.0 * len(shuffle_doc_words_list) / word_doc_freq[vocab[word_id]])
        weight.append(freq * idf)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    return adj


def save_objects(x, y, tx, ty, allx, ally, adj, dataset):
    logger.info("Saving objects.")

    with open(f"data/ind.{dataset}.x", 'wb') as f:
        pkl.dump(x, f)

    with open(f"data/ind.{dataset}.y", 'wb') as f:
        pkl.dump(y, f)

    with open(f"data/ind.{dataset}.tx", 'wb') as f:
        pkl.dump(tx, f)

    with open(f"data/ind.{dataset}.ty", 'wb') as f:
        pkl.dump(ty, f)

    with open(f"data/ind.{dataset}.allx", 'wb') as f:
        pkl.dump(allx, f)

    with open(f"data/ind.{dataset}.ally", 'wb') as f:
        pkl.dump(ally, f)

    with open(f"data/ind.{dataset}.adj", 'wb') as f:
        pkl.dump(adj, f)

    logger.info("All objects saved successfully.")


def main():
    global dataset, vocab, word_embeddings_dim, word_vector_map, word_id_map
    dataset = process_arguments()
    doc_name_list, doc_train_list, doc_test_list, doc_content_list = load_dataset(dataset)
    shuffle_doc_name_list, shuffle_doc_words_list, train_ids, test_ids = shuffle_and_split_data(
        doc_name_list, doc_train_list, doc_test_list, doc_content_list, dataset)

    vocab, vocab_size, word_id_map, word_freq = build_vocabulary(shuffle_doc_words_list)
    # word_id_map is now assigned, and accessible globally

    word_doc_list, word_doc_freq = build_word_doc_list(shuffle_doc_words_list)
    label_list = build_label_list(shuffle_doc_name_list)

    # Parameters
    # word_embeddings_dim = 300  # Already defined globally
    # word_vector_map = {}  # Already defined globally; load your word vectors if available

    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size
    test_size = len(test_ids)
    total_size = len(shuffle_doc_words_list)

    # Compute document vectors for training, test, and allx
    logger.info("Computing document vectors for training set.")
    train_indices = list(range(real_train_size))
    train_doc_vectors = compute_document_vectors(shuffle_doc_words_list, train_indices)
    x = build_feature_matrices(train_doc_vectors, train_indices)
    y = build_labels(shuffle_doc_name_list, label_list, train_indices)

    logger.info("Computing document vectors for test set.")
    test_indices = list(range(train_size, train_size + test_size))
    test_doc_vectors = compute_document_vectors(shuffle_doc_words_list, test_indices)
    tx = build_feature_matrices(test_doc_vectors, test_indices)
    ty = build_labels(shuffle_doc_name_list, label_list, test_indices)

    # Compute allx and ally
    logger.info("Computing allx and ally.")
    allx_indices = list(range(train_size + vocab_size))
    # Combine train_doc_vectors and word_vectors
    all_doc_vectors = train_doc_vectors.copy()
    for i in tqdm(range(vocab_size), desc="Processing vocabulary for allx"):
        idx = train_size + i
        word = vocab[i]
        all_doc_vectors[idx] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)
        if word in word_vector_map:
            all_doc_vectors[idx] = word_vector_map[word]
    allx = build_feature_matrices(all_doc_vectors, list(all_doc_vectors.keys()))
    ally_labels = build_labels(shuffle_doc_name_list, label_list, list(range(train_size)))
    ally = np.vstack([ally_labels, np.zeros((vocab_size, len(label_list)))])

    # Build windows for PMI calculation
    windows = build_windows(shuffle_doc_words_list)
    word_window_freq = compute_word_window_freq(windows)
    word_pair_count = compute_word_pair_count(windows)

    # Build adjacency matrix
    adj = build_adjacency_matrix(train_size, vocab_size, test_size, word_pair_count, word_window_freq, word_doc_freq, windows, shuffle_doc_words_list)

    # Save all objects
    save_objects(x, y, tx, ty, allx, ally, adj, dataset)


if __name__ == '__main__':
    main()
