#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    
    m_s = max([len(s) for s in sents])
    sents_padded = [[w for w in s] + [pad_token] * (m_s - len(s)) for s in sents]
    
    ### END YOUR CODE

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def get_sents_lenth(source, sbol):

    if type(source[0]) is not list:
        source = [source]
    
    source_lengths = [len(s) for s in source]
    XX = [list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]])) for s in source]
    to_add = [[i+1 for i,k in enumerate(s) if k == sbol[1]] for s in source]
    XX = [sorted(list(set(XX[i] + to_add[i]+[source_lengths[i]]))) for i in range(len(XX))]   #len(XX): Batch size
    to_sub = [[i for i,x in enumerate(xx) if x in to_add[j]] for j, xx in enumerate(XX)]
    XX = [[s[i]-s[i-1] if i>0 else s[i] for i in range(len(s))] for s in XX]     # index to interval lenth(어절의 길이)
    XX_len = [len(s) for s in XX]    # 문장의 어절 갯수
    XX_subtracted = [[x-1 if i in to_sub[j] else x for i,x in enumerate(xx)] for j, xx in enumerate(XX)]
    return XX_len, XX, XX_subtracted

def get_sent_lenth(s,sbol):
    
    X = list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]]))
    to_add = [i+1 for i,k in enumerate(s) if k == sbol[1]]
    X = sorted(list(set(X + to_add+[len(s)])))
    #to_sub = [i for i,x in enumerate(X) if x in to_add] 
    X = [X[i]-X[i-1] if i>0 else X[i] for i in range(len(X))]
    #X = [x-1 if i in to_sub else x for i,x in enumerate(X)]
    return sum(X) - len(to_add)


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    
    sbol = [['(', ')', ',', "'", '"'],'_']
    
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        #examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        examples = sorted(examples, key=lambda e: get_sent_lenth(e[0],sbol), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

