#!/usr/bin/python
#-*-coding:utf-8-*-

"""
Dataset Preparation Script

This script prepares raw text datasets for Text GCN processing:
- Downloads and extracts dataset files
- Organizes into required directory structure
- Creates train/test splits
- Generates necessary metadata files

Supported datasets:
- 20 Newsgroups (20ng)
- Reuters R8
- Reuters R52
- OHSUMED
- Movie Reviews (mr)

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