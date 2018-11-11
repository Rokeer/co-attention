# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-07 16:01:25
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13
import reader
import utils
import keras.backend as K
import numpy as np

logger = utils.get_logger("Prepare data ...")

def prepare_sentence_context_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):

    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts, train_context, text_train), (dev_x, dev_y, dev_prompts, dev_context, text_dev), (test_x, test_y, test_prompts, test_context, text_test), vocab, overal_maxlen, overal_maxnum, context_len, context_num = \
        reader.get_data_context(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)
    train_context, dumb, dumb2 = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len, post_padding=True)
    dev_context, dumb, dumb2 = utils.padding_sentence_sequences(dev_context, dev_y, context_num, context_len,
                                                                  post_padding=True)
    test_context, dumb, dumb2 = utils.padding_sentence_sequences(test_context, test_y, context_num, context_len,
                                                                  post_padding=True)
    del dumb
    del dumb2

    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)


    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = reader.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = reader.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))
    logger.info('  context shape: ' + str(train_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, Y_train, mask_train, train_context, text_train), (X_dev, Y_dev, mask_dev, dev_context, text_dev), (X_test, Y_test, mask_test, test_context, text_test), \
            vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean, context_len, context_num

def prepare_sentence_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):

    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts), (dev_x, dev_y, dev_prompts), (test_x, test_y, test_prompts), vocab, overal_maxlen, overal_maxnum = \
        reader.get_data(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overal_maxnum, overal_maxlen, post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overal_maxnum, overal_maxlen, post_padding=True)

    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)


    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = reader.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = reader.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, Y_train, mask_train), (X_dev, Y_dev, mask_dev), (X_test, Y_test, mask_test), \
            vocab, len(vocab), embedd_matrix, overal_maxlen, overal_maxnum, scaled_train_mean


def prepare_data(datapaths, embedding_path=None, embedding='word2vec', embedd_dim=100, prompt_id=1, vocab_size=0, tokenize_text=True, \
                         to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
    # support char features
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_char_x, train_y, train_prompts), (dev_x, dev_char_x, dev_y, dev_prompts), (test_x, test_char_x, test_y, test_prompts), vocab, char_vocab, overal_maxlen, overal_maxnum, maxcharlen = \
        reader.get_char_data(datapaths, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)

    X_train, C_train, y_train, mask_train = utils.padding_sequences(train_x, train_char_x, train_y, overal_maxnum, overal_maxlen, maxcharlen, post_padding=True)
    X_dev, C_dev, y_dev, mask_dev = utils.padding_sequences(dev_x, dev_char_x, dev_y, overal_maxnum, overal_maxlen, maxcharlen, post_padding=True)
    X_test, C_test, y_test, mask_test = utils.padding_sequences(test_x, test_char_x, test_y, overal_maxnum, overal_maxlen, maxcharlen, post_padding=True)

    if prompt_id:
        train_pmt = np.array(train_prompts, dtype='int32')
        dev_pmt = np.array(dev_prompts, dtype='int32')
        test_pmt = np.array(test_prompts, dtype='int32')

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)


    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = reader.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = reader.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = reader.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = reader.get_model_friendly_scores(train_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))
    logger.info('  train char X shape: ' + str(C_train.shape))
    logger.info('  dev char X shape:   ' + str(C_dev.shape))
    logger.info('  test char X shape:  ' + str(C_test.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' % 
                (str(train_mean), str(train_std), str(scaled_train_mean)))

    if embedding_path:
        embedd_dict, embedd_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = utils.build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    return (X_train, C_train, Y_train, mask_train), (X_dev, C_dev, Y_dev, mask_dev), (X_test, C_test, Y_test, mask_test), \
            vocab, len(vocab), char_vocab, len(char_vocab), embedd_matrix, overal_maxlen, overal_maxnum, maxcharlen, scaled_train_mean