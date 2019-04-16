#获得模型参数个数，即权值阈值个数
import tensorflow as tf
from functools import reduce
from operator import mul

def get_num_params():
    num_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        num_params+=reduce(mul,[dim.value for dim in shape],1)
    return num_params