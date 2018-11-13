
import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
import tensorflow
# from keras import activations, initializations, regularizers
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf
from matrix_attention import MatrixAttention
from masked_softmax import MaskedSoftmax
from weighted_sum import WeightedSum
# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
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
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
        self.built = True
    
    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])
                # weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])

        weights = K.softmax(weights)

        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            # out = out.sum(axis=1)
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Build attention pooling layer
class CoAttention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(CoAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # init_val_v = (np.random.randn(input_shape[1]-self.splitpoint) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        # init_val_W = (np.random.randn(input_shape[0][2], input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # self.att_W = K.variable(init_val_W, name='att_W')
        # init_val_cW = (np.random.randn(input_shape[1][2], input_shape[1][2]) * self.init_stdev).astype(K.floatx())
        # self.att_cW = K.variable(init_val_cW, name='att_cW')
        # init_val_oW = (np.random.randn(input_shape[0][2], input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # self.att_oW = K.variable(init_val_oW, name='att_oW')
        # self.trainable_weights = [self.att_v, self.att_W]
        self.trainable_weights = [self.att_v]

    def call(self, x, mask=None):
        x1 = x[0]  # essay
        x2 = x[1]  # context
        # x1 = x[:, :self.splitpoint, :]
        # x2 = x[:, self.splitpoint:, :]

        matrix_attention_layer = MatrixAttention(name='essay_context_similarity')
        essay_context_similarity = matrix_attention_layer([x1, x2])
        essay_context_attention = MaskedSoftmax()(essay_context_similarity)
        weighted_sum_layer = WeightedSum(name="essay_context_vectors", use_masking=False)
        y = weighted_sum_layer([x2, essay_context_attention])
        # weights = K.sum(essay_context_attention, 2)

        # y = K.dot(x1, self.att_W)
        # cy = K.dot(x2, self.att_cW)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, y, axes=[[0], [2]])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
                # weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])

        weights = K.softmax(weights)

        out = x1 * K.permute_dimensions(K.repeat(weights, x1.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            # out = out.sum(axis=1)
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Build attention pooling layer
class CoAttentionWithoutBi(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(CoAttentionWithoutBi, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # init_val_v = (np.random.randn(input_shape[1]-self.splitpoint) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        # init_val_W = (np.random.randn(input_shape[0][2], input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # self.att_W = K.variable(init_val_W, name='att_W')
        # init_val_cW = (np.random.randn(input_shape[1][2], input_shape[1][2]) * self.init_stdev).astype(K.floatx())
        # self.att_cW = K.variable(init_val_cW, name='att_cW')
        # init_val_oW = (np.random.randn(input_shape[0][2], input_shape[0][2]) * self.init_stdev).astype(K.floatx())
        # self.att_oW = K.variable(init_val_oW, name='att_oW')
        # self.trainable_weights = [self.att_v, self.att_W]
        self.trainable_weights = [self.att_v]

    def call(self, x, mask=None):
        x1 = x[0]
        y = x[1]

        # y = K.dot(x1, self.att_W)
        # cy = K.dot(x2, self.att_cW)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, y, axes=[[0], [2]])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
                # weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])

        weights = K.softmax(weights)

        out = x1 * K.permute_dimensions(K.repeat(weights, x1.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            # out = out.sum(axis=1)
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

