#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/2 下午1:46
@Author  : YuJun
@FileName: tile.py
"""

import tensorflow as tf

# tf.tile()应用于需要张量扩展的场景，具体说来就是：
# 如果现有一个形状如[width, height]的张量，需要得到一个基于原张量的，
# 形状如[batch_size,width,height]的张量，其中每一个batch的内容都和原张量一模一样。

# tf.tile使用方法如：
# tile(
#     input,
#     multiples,
#     name=None
# )

if __name__ == "__main__":

    a = tf.constant([7, 19])
    a1 = tf.tile(a, multiples=[3])  # 第一个维度扩充3遍
    b = tf.constant([[4, 5], [3, 5]])
    b1 = tf.tile(b, multiples=[2, 3])  # 第一个维度扩充2遍，第二个维度扩充3遍

    print(a1, '\n', b1)
    # tf.Tensor([ 7 19  7 19  7 19], shape=(6,), dtype=int32)

    #  tf.Tensor(
    # [[4 5 4 5 4 5]
    #  [3 5 3 5 3 5]
    #  [4 5 4 5 4 5]
    #  [3 5 3 5 3 5]], shape=(4, 6), dtype=int32)


