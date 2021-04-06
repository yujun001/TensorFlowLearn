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

# 数据处理过后的一个SparseTensor,
# indices是数组中非0元素下标，values跟indices一一对应，表示该下标位置的值，
def sparse_from_csv(csv):
    """
    :param csv:
    :return:
    """
    ids, post_tags_str = tf.io.decode_csv(csv, [[-1], [""]])
    # output: tf.Tensor([b'harden|james|curry' b'wrestbrook|harden|durant' b'|paul|towns'], shape=(3,), dtype=string)
    # post_tags_str = list(csv)  # 或者
    # output: ['1,harden|james|curry', '2,wrestbrook|harden|durant', '3,|paul|towns']

    table = tf.lookup.StaticHashTable(  # 这里构造了个查找表
        tf.lookup.KeyValueTensorInitializer(keys=tf.constant(TAG_SET),
                                            values=tf.constant(list(range(len(TAG_SET))))), default_value=0)
    split_tags = tf.strings.split(post_tags_str, "|")
    split_tags = tf.RaggedTensor.to_sparse(split_tags)
    return tf.SparseTensor(
        indices=split_tags.indices,
        values=table.lookup(split_tags.values),  # 这里给出了不同值通过表查到的index
        dense_shape=split_tags.dense_shape)


if __name__ == "__main__":

    # ------------------------------------------------------------
    # sparseTensor 构造
    # ------------------------------------------------------------
    example = tf.SparseTensor(indices=[[0], [1], [2]], values=[3, 6, 9], dense_shape=[3])
    example2 = tf.SparseTensor(indices=[[0, 1], [1, 2], [2, 3]], values=[2, 5, 7], dense_shape=[3, 4])
    print("origin sparse example is: \n", example, example2)

    # ------------------------------------------------------------
    # sparse to densor
    # ------------------------------------------------------------
    spare_to_dense = tf.sparse.to_dense(example2)
    print("dense example is: \n ", spare_to_dense)

    # ------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------
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
    print("final embedding:", get_embedding1, '\n', get_embedding2)

    # ------------------------------------------------------------
    # 多值离散: 所谓多值，就是一条数据会拥有该属性多个值，而非一个
    # 将一个field (变量)的特征转换为定长的embedding, 该变量有多个取值, 要变换成定长的embedding。
    # ------------------------------------------------------------
    csv = ["1,harden|james|curry",
           "2,wrestbrook|harden|durant",
           "3,|paul|towns",
           "4,|paul|towns"]
    TAG_SET = ["harden", "james", "curry", "durant", "paul", "towns", "wrestbrook"]

    # 定义embedding变量
    TAG_EMBEDDING_DIM = 10
    embedding_params = tf.Variable(tf.random.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))

    # 得到embedding值; sp_ids就是我们刚刚得到的SparseTensor,
    # sp_weights = None代表每一个取值权重,None的话, 权重是1, 相当于取了平均。
    tags = sparse_from_csv(csv)
    embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None, combiner=None)
    print("final embedding:\n ", embedded_tags)





