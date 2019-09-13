# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 12:57 PM

from __future__ import absolute_import, division, print_function, unicode_literals
import reader
import utils

logger = utils.get_logger("Prepare data ...")


def prepare_sentence_data(
        datapaths,
        embedding_path=None,
        embedding='word2vec',
        emb_dim=100,
        prompt_id=1,
        vocab_size=0,
        tokenize_text=True,
        to_lower=True,
        vocab_path=None,
        score_index=6,
        need_context=True
):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y, train_prompts, train_text), \
    (dev_x, dev_y, dev_prompts, dev_text), \
    (test_x, test_y, test_prompts, test_text), \
    vocab, overall_maxlen, overall_maxnum = \
        reader.get_data(
            datapaths,
            prompt_id,
            vocab_size,
            tokenize_text,
            to_lower,
            vocab_path,
            score_index)

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overall_maxnum, overall_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overall_maxnum, overall_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overall_maxnum, overall_maxlen,
                                                                 post_padding=True)

    if need_context:
        context, context_len, context_num = reader.get_context(prompt_id, vocab, to_lower)
    else:
        # Dummy context
        context = [[0]]
        context_len = 1
        context_num = 1
    train_context = [context] * len(train_x)
    dev_context = [context] * len(dev_x)
    test_context = [context] * len(test_x)

    train_context, _, _ = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len, post_padding=True)
    dev_context, _, _ = utils.padding_sentence_sequences(dev_context, dev_y, context_num, context_len, post_padding=True)
    test_context, _, _ = utils.padding_sentence_sequences(test_context, test_y, context_num, context_len, post_padding=True)

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
    Y_train = utils.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = utils.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = utils.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = utils.get_model_friendly_scores(train_mean, prompt_id)
    scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, prompt_id)
    scaled_test_mean = utils.get_model_friendly_scores(test_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    if need_context:
        logger.info('  train context shape: ' + str(train_context.shape))
        logger.info('  dev context shape: ' + str(dev_context.shape))
        logger.info('  test context shape: ' + str(test_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    if embedding_path:
        emb_dict, emb_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, emb_dim)
        emb_matrix = utils.build_embedding_table(vocab, emb_dict, emb_dim, logger, caseless=True)
    else:
        emb_matrix = None

    return (X_train, Y_train, mask_train, train_context, train_text), \
           (X_dev, Y_dev, mask_dev, dev_context, dev_text), \
           (X_test, Y_test, mask_test, test_context, test_text), \
           vocab, len(vocab), emb_matrix, overall_maxlen, overall_maxnum, scaled_train_mean, context_len, context_num
