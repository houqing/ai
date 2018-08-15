from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def main() :
    x = np.fromfile("11--110--resnet_model-res2a_relu-0--float16.fmout.bin", dtype=np.float16).reshape(1,64,56,56)
    y_dumped = np.fromfile("11--128--resnet_model-res2a_branch1-Conv2D-0--float16.fmout.bin", dtype=np.float16).reshape(1,256,56,56)
    w  = np.fromfile("11--124--resnet_model-res2a_branch1-kernel_cast-0--float16.var_cast.bin", dtype=np.float16).reshape(1,1,64,256)

    strides = [1, 1, 1, 1]
    padding = 'SAME'

    y_true = tf.nn.conv2d(x, w, strides, padding, data_format='NCHW')

    result_out = None
    with tf.Session():
        result_out = y_true.eval()

        result_out.astype(np.float16).tofile("test_result_out_fp16.bin")

        z = y_dumped - result_out
        n = tf.count_nonzero(z).eval()
        print("========")
        if n == 0:
            print("PASS: data just generated is the same as reference data")
        else:
            print("FAIL: data just generated is different than reference data, different count is [" + n + "]")
        print("========")

if __name__ == '__main__':
    main()

