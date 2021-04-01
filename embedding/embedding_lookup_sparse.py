#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2021/3/31 下午1:50
@Author  : YuJun
@FileName: embedding_lookup_sparse.py
"""

# SparseTensor 稀疏张量

import tensorflow as tf
import numpy as np
import pandas as pd

example = tf.SparseTensor(indices=[[0], [1], [2]], values=[3, 6, 9], dense_shape=[3])
example2 = tf.SparseTensor(indices=[[0, 1], [1, 2], [2, 3]], values=[2, 5, 7], dense_shape=[3, 4])

# 稀疏乘法
# tf.sparse.sparse_dense_matmul()

embedding = tf.constant([[0.21, 0.41, 0.51, 0.11],
                         [0.22, 0.42, 0.52, 0.12],
                         [0.23, 0.43, 0.53, 0.13],
                         [0.24, 0.44, 0.54, 0.14]], dtype=tf.float32)
feature_batch = tf.constant([2, 3, 1, 0])

# 1) matmul
feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)
get_embedding1 = tf.matmul(feature_batch_one_hot, embedding)

# 2) embedding_lookup
get_embedding2 = tf.nn.embedding_lookup(embedding, feature_batch)


if __name__ == "__main__":

    # origin sparse data
    print("sparse example: ", example2)

    # sparse to dense
    spare_to_dense = tf.sparse.to_dense(example2)
    print("dense example: ", spare_to_dense)

    # embedding matmul
    print(get_embedding1)
    print(get_embedding2)










