#!/usr/bin/python
#-*-coding:utf-8-*-

"""
Dataset Preparation Script for Text GCN

This script handles the preparation and preprocessing of text datasets for GCN analysis.
It supports multiple standard datasets and can be extended for custom data.

Key Features:
- Flexible dataset handling
- Train/test split generation
- Metadata creation
- Corpus organization

Supported Datasets:
- 20 Newsgroups (20ng)
- Reuters R8 and R52
- OHSUMED
- Movie Reviews (mr)
- Custom datasets

File Structure Created:
data/
├── dataset_name.txt        # Metadata file
└── corpus/
    └── dataset_name.txt    # Actual corpus content

Usage:
    python prepare_data.py <dataset>
"""

dataset_name = 'own'
sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
labels = ['Yes' , 'No' ]
train_or_test_list = ['train', 'test']


meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()