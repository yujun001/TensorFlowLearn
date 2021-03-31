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

if __name__ == "__main__":

    # origin sparse data
    print("sparse example: ", example2)

    # sparse to dense
    spare_to_dense = tf.sparse.to_dense(example2)
    print("dense example: ", spare_to_dense)









