# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 2:22 PM

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Reshape, TimeDistributed, Conv1D, LSTM, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop

from networks.zeromasking import ZeroMaskedEntries
from networks.softattention import Attention
from networks.matrix_attention import MatrixAttention
from networks.masked_softmax import MaskedSoftmax
from networks.weighted_sum import WeightedSum
from networks.max import Max
from networks.repeat_like import RepeatLike
from networks.complex_concat import ComplexConcat

from utils import get_logger
import numpy as np
import time

logger = get_logger("Build model")


def get_optimizer(name, lr):
    if name == 'sgd':
        return SGD(lr=lr)
    elif name == 'adagrad':
        return Adagrad(lr=lr)
    elif name == 'rmsprop':
        return RMSprop(lr=lr)
    else:
        raise NotImplementedError


def compile_model(model, opts):
    optimizer = get_optimizer(opts.optimizer, opts.learning_rate)
    start_time = time.time()
    model.compile(loss='mse', optimizer=optimizer)
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)


def build_hrcnn_model(
        opts,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=word_input, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_shrcnn_model(
        opts,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        maxcnum=50,
        maxclen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN with CO-ATTN based on sentence level
    N = maxnum
    L = maxlen

    cN = maxcnum
    cL = maxclen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    context_input = Input(shape=(cN * cL,), dtype='int32', name='context_input')

    emb = Embedding(output_dim=embedd_dim, input_dim=vocab_size, weights=embedding_weights, mask_zero=True, name='cx')
    cx = emb(context_input)
    cx_maskedout = ZeroMaskedEntries(name='cx_maskedout')(cx)
    drop_cx = Dropout(opts.dropout, name='drop_cx')(cx_maskedout)

    resh_C = Reshape((cN, cL, embedd_dim), name='resh_C')(drop_cx)

    czcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='czcnn')(resh_C)

    x = emb(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    # pooling mode
    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
    avg_czcnn = TimeDistributed(Attention(), name='avg_czcnn')(czcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
    chz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='chz_lstm')(avg_czcnn)


    logger.info('Use co-attention on text')

    # PART 2:
    # Now we compute a similarity between the passage words and the question words, and
    # normalize the matrix in a couple of different ways for input into some more layers.
    matrix_attention_layer = MatrixAttention(name='essay_context_similarity')
    # Shape: (batch_size, num_passage_words, num_question_words)
    essay_context_similarity = matrix_attention_layer([hz_lstm, chz_lstm])

    # Shape: (batch_size, num_passage_words, num_question_words), normalized over question
    # words for each passage word.
    essay_context_attention = MaskedSoftmax()(essay_context_similarity)
    weighted_sum_layer = WeightedSum(name="essay_context_vectors", use_masking=False)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    weighted_hz_lstm = weighted_sum_layer([chz_lstm, essay_context_attention])

    # Min's paper finds, for each document word, the most similar question word to it, and
    # computes a single attention over the whole document using these max similarities.
    # Shape: (batch_size, num_passage_words)
    context_essay_similarity = Max(axis=-1)(essay_context_similarity)
    # Shape: (batch_size, num_passage_words)
    context_essay_attention = MaskedSoftmax()(context_essay_similarity)
    # Shape: (batch_size, embedding_dim * 2)
    weighted_sum_layer = WeightedSum(name="context_essay_vector", use_masking=False)
    context_essay_vector = weighted_sum_layer([hz_lstm, context_essay_attention])

    # Then he repeats this question/passage vector for every word in the passage, and uses it
    # as an additional input to the hidden layers above.
    repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    tiled_context_essay_vector = repeat_layer([context_essay_vector, hz_lstm])

    complex_concat_layer = ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
    final_merged_passage = complex_concat_layer([hz_lstm,
                                                 weighted_hz_lstm,
                                                 tiled_context_essay_vector])

    avg_hz_lstm = LSTM(opts.lstm_units, return_sequences=False, name='avg_hz_lstm')(final_merged_passage)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[word_input, context_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    start_time = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)

    return model

