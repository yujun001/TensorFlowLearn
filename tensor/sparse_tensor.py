#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/1 下午4:00
@Author  : YuJun
@FileName: sparse_tensor.py
"""

import tensorflow as tf

# Creating a tf.SparseTensor
st1 = tf.SparseTensor(indices=[[0, 3], [2, 4]],
                      values=[10, 20],
                      dense_shape=[3, 10])

def pprint_sparse_tensor(st):
    """
    :param st:  SparseTensor
    :return:
    """
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())

    return s + "}>"

# Manipulating sparse tensors
st_a = tf.SparseTensor(indices=[[0, 2], [3, 4]],
                       values=[31, 2],
                       dense_shape=[4, 10])

st_b = tf.SparseTensor(indices=[[0, 2], [7, 0]],
                       values=[56, 38],
                       dense_shape=[4, 10])

st_c = tf.SparseTensor(indices=([0, 1], [1, 0], [1, 1]),
                       values=[13, 15, 17],
                       dense_shape=[2, 2])

sparse_pattern_A = tf.SparseTensor(indices = [[2,4], [3,3], [3,4], [4,3], [4,4], [5,4]],
                                   values = [1,1,1,1,1,1],
                                   dense_shape = [8,5])
sparse_pattern_B = tf.SparseTensor(indices = [[0,2], [1,1], [1,3], [2,0], [2,4], [2,5], [3,5],
                                              [4,5], [5,0], [5,4], [5,5], [6,1], [6,3], [7,2]],
                                   values = [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                   dense_shape = [8,6])
sparse_pattern_C = tf.SparseTensor(indices = [[3,0], [4,0]],
                                   values = [1,1],
                                   dense_shape = [8,6])

# // start 切片 左闭右开, 范围内不选
sparse_slice_A = tf.sparse.slice(sparse_pattern_A, start = [0,0], size = [8,5])
sparse_slice_B = tf.sparse.slice(sparse_pattern_B, start = [0,5], size = [8,6])
sparse_slice_C = tf.sparse.slice(sparse_pattern_C, start = [0,10], size = [8,6])

if __name__ == "__main__":

    print(pprint_sparse_tensor(st1))

    # 1) construct sparse tensors from dense tensor
    st2 = tf.sparse.from_dense([[1, 0, 0, 8],
                                [0, 0, 0, 0],
                                [0, 0, 3, 0]])
    print(pprint_sparse_tensor(st2))
    st2_dense = tf.sparse.to_dense(st2)
    print("st2 sparse tensor :", st2)
    print("st2_dense tensor is :", st2_dense)

    # 2) Add sparse tensors of the same shape
    st_sum = tf.sparse.add(st_a, st_b)
    print(pprint_sparse_tensor(st_sum))

    # 3) multiply sparse tensors with dense matrices.
    mb = tf.constant([[4], [6]])
    product = tf.sparse.sparse_dense_matmul(st_c, mb)
    print(product)

    # 4) Put sparse tensors together
    sparse_patterns_list = [sparse_pattern_A, sparse_pattern_B, sparse_pattern_C]
    sparse_pattern = tf.sparse.concat(axis=1, sp_inputs=sparse_patterns_list)
    print(tf.sparse.to_dense(sparse_pattern))

    # 5）slice 切片
    print(tf.sparse.to_dense(sparse_pattern_A))
    print(tf.sparse.to_dense(sparse_slice_A))
    print(tf.sparse.to_dense(sparse_slice_B))
    print(tf.sparse.to_dense(sparse_slice_C))  # 超出范围, 不选择

    # 6）elementwise operations on nonzero values in sparse tensors.
    # st2_plus_5 = tf.sparse.map_values(tf.add, st2, 5)
    # print(tf.sparse.to_dense(st2_plus_5))   # only the nonzero values were modified

    # // 只有非0 值被修改
    st2_plus_5 = tf.SparseTensor(st2.indices,
                                 st2.values + 5,
                                 st2.dense_shape)
    print(tf.sparse.to_dense(st2_plus_5))







