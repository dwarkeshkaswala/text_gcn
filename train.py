import time
import tensorflow as tf
from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import sys
import numpy as np
import argparse

# tf.debugging.set_log_device_placement(True)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Dataset string.')
parser.add_argument('--model', default='gcn', help='Model string: gcn, gcn_cheby, dense.')
parser.add_argument('--learning_rate', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=200, help='Number of units in hidden layer 1.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', type=int, default=10, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--featureless', action='store_true', help='Use featureless input')

args = parser.parse_args()

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
if args.dataset not in datasets:
    sys.exit("Wrong dataset name")

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    args.dataset)
print(adj)
features = sp.identity(features.shape[0])  # Featureless
args.featureless = True  # Manually set featureless to True

print("Adjacency matrix shape:", adj.shape)
print("Feature matrix shape:", features.shape)

# Some preprocessing
features = preprocess_features(features)
if args.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif args.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, args.max_degree)
    num_supports = 1 + args.max_degree
    model_func = GCN
elif args.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used in MLP
    num_supports = 1
    model_func = MLP
    args.featureless = False  # For MLP, features are not featureless
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))

# Convert data to appropriate TensorFlow formats
# Features as SparseTensor
features = tf.sparse.SparseTensor(
    indices=np.array(features[0], dtype=np.int64),
    values=features[1].astype(np.float32),
    dense_shape=features[2]
)

# Convert support to SparseTensor with float32 values
if isinstance(support, list):
    support = [tf.sparse.SparseTensor(
        indices=np.array(sup[0], dtype=np.int64),
        values=sup[1].astype(np.float32),
        dense_shape=sup[2]
    ) for sup in support]
else:
    support = [tf.sparse.SparseTensor(
        indices=np.array(support[0], dtype=np.int64),
        values=support[1].astype(np.float32),
        dense_shape=support[2]
    )]

# Labels and masks as tensors
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
train_mask = tf.convert_to_tensor(train_mask)
val_mask = tf.convert_to_tensor(val_mask)
test_mask = tf.convert_to_tensor(test_mask)

# Create model
model = model_func(input_dim=features.shape[1],
                   output_dim=y_train.shape[1],
                   support=support,  # Pass support here
                   args=args,
                   name='gcn_model',
                   logging=True)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# Define training step
@tf.function
def train_step(features, labels, mask):
    with tf.GradientTape() as tape:
        logits = model(features, training=True)
        loss_value = model.compute_loss(features, labels, mask)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    accuracy = model.compute_accuracy(features, labels, mask)
    return loss_value, accuracy

# Define evaluation function
def evaluate(features, labels, mask):
    logits = model(features, training=False)
    loss_value = model.compute_loss(features, labels, mask)
    accuracy = model.compute_accuracy(features, labels, mask)
    return loss_value, accuracy, logits

cost_val = []

# Train model
for epoch in range(args.epochs):
    t = time.time()
    # Training step
    loss_value, acc = train_step(features, y_train, train_mask)

    # Validation
    val_loss, val_acc, _ = evaluate(features, y_val, val_mask)


    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(loss_value.numpy()),
          "train_acc=", "{:.5f}".format(acc.numpy()),
          "val_loss=", "{:.5f}".format(val_loss.numpy()),
          "val_acc=", "{:.5f}".format(val_acc.numpy()),
          "time=", "{:.5f}".format(time.time() - t))

    cost_val.append(val_loss.numpy())

    # Early stopping
    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_loss, test_acc, logits = evaluate(features, y_test, test_mask)
print("Test set results:",
      "cost=", "{:.5f}".format(test_loss.numpy()),
      "accuracy=", "{:.5f}".format(test_acc.numpy()))

# Get predictions and labels for test data
preds = tf.argmax(logits, axis=1)
labels = tf.argmax(y_test, axis=1)

test_mask_bool = test_mask.numpy().astype(bool)
test_pred = preds.numpy()[test_mask_bool]
test_labels = labels.numpy()[test_mask_bool]

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# Extract embeddings
embeddings = model.embedding.numpy()

# Split embeddings
train_size = y_train.shape[0]
test_size = y_test.shape[0]
total_size = embeddings.shape[0]
vocab_size = total_size - train_size - test_size

train_doc_embeddings = embeddings[:train_size]
word_embeddings = embeddings[train_size:train_size + vocab_size]
test_doc_embeddings = embeddings[train_size + vocab_size:]

print('Embeddings shapes:')
print('Word embeddings:', word_embeddings.shape)
print('Train doc embeddings:', train_doc_embeddings.shape)
print('Test doc embeddings:', test_doc_embeddings.shape)

# Save word embeddings
with open('data/corpus/' + args.dataset + '_vocab.txt', 'r') as f:
    words = f.readlines()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open('data/' + args.dataset + '_word_vectors.txt', 'w') as f:
    f.write(word_embeddings_str)

# Save document embeddings
doc_vectors = []
doc_id = 0
for i in range(train_doc_embeddings.shape[0]):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_doc_embeddings.shape[0]):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + args.dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)
