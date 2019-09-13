# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 8:53 PM

from __future__ import absolute_import, division, print_function, unicode_literals
"""
These are utility functions that are similar to calls to Keras' backend.  Some of these are here
because a current function in keras.backend is broken, some are things that just haven't been
implemented.
"""
import tensorflow.keras.backend as K
import tensorflow as tf

VERY_LARGE_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_NEGATIVE_NUMBER = -VERY_LARGE_NUMBER


def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only
    accepts scalar value conditions, rather than boolean tensors which are treated in an
    elementwise function.  This doesn't match with Theano's implementation of switch, but using
    tensorflow's where, we can exactly retrieve this functionality.
    """

    cond_shape = cond.get_shape()
    input_shape = then_tensor.get_shape()
    if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
        # This happens when the last dim in the input is an embedding dimension. Keras usually does not
        # mask the values along that dimension. Theano broadcasts the value passed along this dimension,
        # but TF does not. Using K.dot() since cond can be a tensor.
        cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
    return tf.where(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)


def very_negative_like(tensor):
    return K.ones_like(tensor) * VERY_NEGATIVE_NUMBER


def last_dim_flatten(input_tensor):
    '''
    Takes a tensor and returns a matrix while preserving only the last dimension from the input.
    '''
    input_ndim = K.ndim(input_tensor)
    shuffle_pattern = (input_ndim - 1,) + tuple(range(input_ndim - 1))
    dim_shuffled_input = K.permute_dimensions(input_tensor, shuffle_pattern)
    return K.transpose(K.batch_flatten(dim_shuffled_input))

