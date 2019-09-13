# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 12:45 PM

from __future__ import absolute_import, division, print_function, unicode_literals

from gensim.models import KeyedVectors
from score_ranges import score_ranges

import gzip
import logging
import sys
import numpy as np
import tensorflow as tf


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def padding_sentence_sequences(index_sequences, scores, max_sentnum, max_sentlen, post_padding=True):

    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen], dtype=np.float32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # X_len[i] = length
            for k in range(length):
                wid = word_ids[k]
                # print wid
                X[i, j, k] = wid

            # Zero out X after the end of the sequence
            X[i, j, length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, j, :length] = 1

        X[i, num:, :] = 0
        Y[i] = scores[i]
    return X, Y, mask


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # loading Senna
        logger.info("Loading Senna ...")
        embedd_dim = -1
        embedd_dict = dict()
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'random':
        # loading random embedding table
        logger.info("Loading Random ...")
        embedd_dict = dict()
        words = word_alphabet.get_content()
        scale = np.sqrt(3.0 / embedd_dim)
        # print words, len(words)
        for word in words:
            embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
        return embedd_dict, embedd_dim, False
    else:
        raise ValueError("embedding should choose from [word2vec, glove, senna, random]")


def build_embedding_table(word_alphabet, emb_dict, emb_dim, logger, caseless):
    scale = np.sqrt(3.0 / emb_dim)
    emb_table = np.empty([len(word_alphabet), emb_dim], dtype=np.float32)
    emb_table[0, :] = np.zeros([1, emb_dim])
    oov_num = 0
    for word, index in word_alphabet.items():
        ww = word.lower() if caseless else word
        # show oov ratio
        if ww in emb_dict:
            emb = emb_dict[ww]
        else:
            emb = np.random.uniform(-scale, scale, [1, emb_dim])
            oov_num += 1
        emb_table[index, :] = emb
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    logger.info("OOV number = %s, OOV ratio = %f" % (oov_num, oov_ratio))
    return emb_table


def get_model_friendly_scores(scores_array, prompt_id):
    low, high = score_ranges[prompt_id]
    scores_array = (scores_array - low) / (high - low)
    assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scaled_scores, prompt_id):
    arg_type = type(prompt_id)
    assert arg_type in {int, np.ndarray}

    low, high = score_ranges[prompt_id]
    scores_array = scaled_scores * (high - low) + low
    assert np.all(scores_array >= low) and np.all(scores_array <= high)

    return np.around(scores_array).astype(int)
