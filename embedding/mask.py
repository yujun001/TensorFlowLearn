#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/2 下午12:45
@Author  : YuJun
@FileName: mask.py
"""

import tensorflow as tf
import numpy as np

raw_inputs = [[711, 632, 71],
              [73, 8, 3215, 55, 927],
              [83, 91, 1, 645, 1253, 927],
              ]

class MyLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(32)

    def call(self, inputs):
        x = self.embedding(inputs)
        # Note that you could also prepare a `mask` tensor manually.
        # It only needs to be a boolean tensor
        # with the right shape, i.e. (batch_size, timesteps).
        mask = self.embedding.compute_mask(inputs)
        output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
        return output

class TemporalSplit(tf.keras.layers.Layer):
    """Split the input tensor into 2 tensors along the time dimension."""
    def call(self, inputs):
        # Expect the input to be 3D and mask to be 2D, split the input tensor into 2
        # subtensors along the time axis (axis 1).
        return tf.split(inputs, 2, axis=1)

    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 2 if it presents.
        if mask is None:
            return None
        return tf.split(mask, 2, axis=1)

class CustomEmbedding(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            dtype="float32",
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

class MyActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyActivation, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)

class TemporalSoftmax(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        inputs_exp = tf.exp(inputs) * broadcast_float_mask
        inputs_sum = tf.reduce_sum(inputs * broadcast_float_mask, axis=1, keepdims=True)
        return inputs_exp / inputs_sum


if __name__ == "__main__":

    # 1) 截断和填充 列表, post后, pre前, value 填充值
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sequences=raw_inputs,
                                                                  maxlen=6,
                                                                  padding="post", value=0)
    print(padded_inputs)

    # 2) 遮盖 mask
    # 所有样本都具有了统一长度，必须告知模型，数据的某些部分实际上是填充，应该忽略。这种机制就是遮盖。
    embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
    masked_output = embedding(padded_inputs)
    print(masked_output._keras_mask)

    # Simulate the embedding lookup by expanding the 2D input to 3D, with embedding dimension of 10.
    masking_layer = tf.keras.layers.Masking()
    unmasked_embedding = tf.cast(
        tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32)
    masked_embedding = masking_layer(unmasked_embedding)
    print(masked_embedding._keras_mask)

    # 每个 False 条目表示对应的时间步骤应在处理时忽略

    # 3）Mask propagation in the Functional API and Sequential API
    # Sequential API
    model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True),
         tf.keras.layers.LSTM(32), ])

    # Functional API
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
    outputs = tf.keras.layers.LSTM(32)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    # 4) Passing mask tensors directly to layers
    # 将掩码张量直接传递给层
    layer = MyLayer()
    x = np.random.random((32, 10)) * 100
    x = x.astype("int32")
    print(x)
    print(layer(x))

    # 5) Supporting masking in your custom layers
    # 5.1
    first_half, second_half = TemporalSplit()(masked_embedding)
    print(first_half._keras_mask)
    print(second_half._keras_mask)

    # 5.2 能够根据输入值生成掩码：
    layer = CustomEmbedding(10, 32, mask_zero=True)
    x = np.random.random((3, 10)) * 9
    x = x.astype("int32")
    y = layer(x)
    mask = layer.compute_mask(x)
    print(mask)

    # 5.3 在掩码生成层（如 Embedding）和掩码使用层（如 LSTM）之间使用此自定义层，它会将掩码一路传递到掩码使用层。
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
    x = MyActivation()(x)  # Will pass the mask along
    print("Mask found:", x._keras_mask)
    outputs = tf.keras.layers.LSTM(32)(x)  # Will receive the mask
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    # 5.4
    # 示例中的层在输入序列的时间维度（轴 1）上计算 Softmax，同时丢弃遮盖的时间步骤。
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=10, output_dim=32, mask_zero=True)(inputs)
    x = tf.keras.layers.Dense(1)(x)
    outputs = TemporalSoftmax()(x)

    model = tf.keras.Model(inputs, outputs)
    y = model(np.random.randint(0, 10, size=(32, 100)), np.random.random((32, 100, 1)))

    print(np.random.randint(0, 10, size=(32, 100)))
    print(np.random.random((32, 100, 1)))

    # padding and masking
    # "Masking" is how layers are able to know when to skip / ignore certain timesteps in sequence inputs.
    # Some layers are mask-generators: Embedding can generate a mask from input values (if mask_zero=True), and so can the Masking layer.
    # Some layers are mask-consumers: they expose a mask argument in their __call__ method. This is the case for RNN layers.
    # In the Functional API and Sequential API, mask information is propagated automatically.
    # When using layers in a standalone way, you can pass the mask arguments to layers manually.
    # You can easily write layers that modify the current mask, that generate a new mask, or that consume the mask associated with the inputs.

