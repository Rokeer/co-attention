# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 2:57 PM

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np


class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = tf.Variable(init_val_v, name='att_v', trainable=True)
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = tf.Variable(init_val_W, name='att_W', trainable=True)

    def call(self, inputs, mask=None):
        y = K.dot(inputs, self.att_W)
        if not self.activation:
            weights = tf.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])

        weights = K.softmax(weights)

        out = inputs * K.permute_dimensions(K.repeat(weights, inputs.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            # out = out.sum(axis=1)
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))