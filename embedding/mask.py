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
                                                                  maxlen=5,
                                                                  padding="post", value=1000)
    print(padded_inputs)
    print(raw_inputs)
