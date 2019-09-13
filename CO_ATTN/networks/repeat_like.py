# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/12/19 10:08 PM

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer
import tensorflow as tf


class RepeatLike(Layer):
    """
    This ``Layer`` is like :class:`~.repeat.Repeat`, but gets the number of repetitions to use from
    a second input tensor.  This allows doing a number of repetitions that is unknown at graph
    compilation time, and is necessary when the ``repetitions`` argument to ``Repeat`` would be
    ``None``.

    If the mask is not ``None``, we must be able to call ``K.expand_dims`` using the same axis
    parameter as we do for the input.

    Input:
        - A tensor of arbitrary shape, which we will expand and tile.
        - A second tensor whose shape along one dimension we will copy

    Output:
        - The input tensor repeated along one of the dimensions.

    Parameters
    ----------
    axis: int
        We will add a dimension to the input tensor at this axis.
    copy_from_axis: int
        We will copy the dimension from the second tensor at this axis.
    """
    def __init__(self, axis: int, copy_from_axis: int, **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.copy_from_axis = copy_from_axis
        super(RepeatLike, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None or mask[0] is None:
            return None
        expanded = tf.expand_dims(mask[0], self.axis)
        ones = [1] * K.ndim(expanded)
        num_repetitions = inputs[1].shape[self.copy_from_axis]
        tile_shape = tf.concat([ones[:self.axis], [num_repetitions], ones[self.axis + 1:]], 0)
        return tf.tile(expanded, tile_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:self.axis] + (input_shape[1][self.copy_from_axis],) + input_shape[0][self.axis:]

    def call(self, inputs, mask=None):
        expanded = tf.expand_dims(inputs[0], self.axis)
        ones = [1] * K.ndim(expanded)
        num_repetitions = inputs[1].shape[self.copy_from_axis]
        tile_shape = tf.concat([ones[:self.axis], [num_repetitions], ones[self.axis + 1:]], 0)
        return tf.tile(expanded, tile_shape)


    def get_config(self):
        base_config = super(RepeatLike, self).get_config()
        config = {'axis': self.axis, 'copy_from_axis': self.copy_from_axis}
        config.update(base_config)
        return config
