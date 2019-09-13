# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 2:43 PM

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.layers import Layer

import tensorflow as tf
import tensorflow.keras.backend as K


class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, mask_zero=False, **kwargs):
        super(ZeroMaskedEntries, self).__init__(**kwargs)
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, inputs, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return inputs * mask

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            return tf.not_equal(inputs, 0)
