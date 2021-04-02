#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/2 下午12:45
@Author  : YuJun
@FileName: mask.py
"""

import tensorflow as tf


raw_inputs = [[711, 632, 71],
              [73, 8, 3215, 55, 927],
              [83, 91, 1, 645, 1253, 927],
              ]

if __name__ == "__main__":

    # 1) 截断和填充 列表, post后, pre前, value 填充值
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(sequences=raw_inputs,
                                                                  maxlen=6,
                                                                  padding="post", value=0)
    print(padded_inputs)
    # print(raw_inputs)

    # 2) 遮盖 mask
    # 所有样本都具有了统一长度，必须告知模型，数据的某些部分实际上是填充，应该忽略。这种机制就是遮盖。
    embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
    masked_output = embedding(padded_inputs)
    print(masked_output._keras_mask)

    masking_layer = tf.keras.layers.Masking()
    # Simulate the embedding lookup by expanding the 2D input to 3D,
    # with embedding dimension of 10.

    print(tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]))

    unmasked_embedding = tf.cast(
        tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32)
    masked_embedding = masking_layer(unmasked_embedding)
    print(masked_embedding._keras_mask)
    # unmasked_embedding = tf.cast(
    #     tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32
    # )
    #
    # masked_embedding = masking_layer(unmasked_embedding)
    # print(masked_embedding._keras_mask)
