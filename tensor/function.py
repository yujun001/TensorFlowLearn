#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/2 下午4:40
@Author  : YuJun
@FileName: function.py
"""
import tensorflow as tf

@tf.function  # The decorator converts `add` into a `Function`.
def add(a, b):
    return a + b

tensor_1 = tf.ones([2, 2])
tensor_2 = tf.ones([2, 2])
tensor_res = tf.add(tensor_1, tensor_2)
# print(tensor_res)  # [[2., 2.], [2., 2.]]

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)