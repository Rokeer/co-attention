# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 1:21 PM

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import time
import numpy as np
from utils import get_logger
import data_prepare

from networks.network_models import build_hrcnn_model, build_shrcnn_model

from evaluator import Evaluator

logger = get_logger("Main")
np.random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN_LSTM model")
    parser.add_argument('--embedding', type=str, default='glove',
                        help='Word embedding type, glove, word2vec, senna or random')
    parser.add_argument('--embedding_dict', type=str, default='glove/glove.6B.50d.txt',
                        help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=50,
                        help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000,
                        help="Vocab size (default=4000)")

    parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    parser.add_argument('--optimizer', choices=['sgd', 'adagrad', 'rmsprop'], help='Optimizer', default='rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='checkpoints')

    parser.add_argument('--train', type=str, help='train file', default='data/fold_0/train.tsv')
    parser.add_argument('--dev', type=str, help='dev file', default='data/fold_0/dev.tsv')
    parser.add_argument('--test', type=str, help='test file', default='data/fold_0/test.tsv')
    parser.add_argument('--prompt_id', type=int, default=3, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true',
                        help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['att', 'co'], default='co',
                        help='attention-pooling, or co-attention pooling')

    args = parser.parse_args()

    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_path
    num_epochs = args.num_epochs

    modelname = "%s.prompt%s.%sfilters.bs%s" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

    datapaths = [args.train, args.dev, args.test]
    embedding_path = args.embedding_dict
    embedding = args.embedding
    emb_dim = args.embedding_dim
    prompt_id = args.prompt_id

    mode = args.mode
    need_context = mode in ['co']

    (X_train, Y_train, mask_train, train_context, text_train), \
    (X_dev, Y_dev, mask_dev, dev_context, text_dev), \
    (X_test, Y_test, mask_test, test_context, text_test), \
    vocab, vocab_size, emb_table, overall_maxlen, overall_maxnum, init_mean_value, context_len, context_num = \
        data_prepare.prepare_sentence_data(
            datapaths,
            embedding_path,
            embedding,
            emb_dim,
            prompt_id,
            args.vocab_size,
            tokenize_text=True,
            to_lower=True,
            vocab_path=None,
            score_index=6,
            need_context=need_context
        )

    if emb_table is not None:
        emb_dim = emb_table.shape[1]
        emb_table = [emb_table]

    max_sentnum = overall_maxnum
    max_sentlen = overall_maxlen

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1] * X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    train_context = train_context.reshape((train_context.shape[0], train_context.shape[1] * train_context.shape[2]))
    dev_context = dev_context.reshape((dev_context.shape[0], dev_context.shape[1] * dev_context.shape[2]))
    test_context = test_context.reshape((test_context.shape[0], test_context.shape[1] * test_context.shape[2]))

    logger.info("X_train shape: %s" % str(X_train.shape))
    logger.info("X_dev shape: %s" % str(X_dev.shape))
    logger.info("X_test shape: %s" % str(X_test.shape))

    if mode == 'att':
        model = build_hrcnn_model(args, vocab_size, max_sentnum, max_sentlen, emb_dim, emb_table, True, init_mean_value)
        x_train = X_train
        y_train = Y_train
        x_dev = X_dev
        y_dev = Y_dev
        x_test = X_test
        y_test = Y_test
    elif mode == 'co':
        model = build_shrcnn_model(args, vocab_size, max_sentnum, max_sentlen, context_num, context_len, emb_dim, emb_table, True, init_mean_value)
        x_train = [X_train, train_context]
        y_train = Y_train
        x_dev = [X_dev, dev_context]
        y_dev = Y_dev
        x_test = [X_test, test_context]
        y_test = Y_test
    else:
        raise NotImplementedError

    evl = Evaluator(
        prompt_id,
        checkpoint_dir,
        modelname,
        x_train,
        x_dev,
        x_test,
        y_train,
        y_dev,
        y_test
    )

    # Initial evaluation
    logger.info("Initial evaluation: ")
    evl.evaluate(model, -1, print_info=True)
    logger.info("Train model")
    for ii in range(num_epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), num_epochs))
        start_time = time.time()
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evl.evaluate(model, ii + 1, print_info=True)

    evl.print_final_info()


if __name__ == '__main__':
    main()
