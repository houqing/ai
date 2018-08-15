from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def main() :
    dy = np.fromfile("11--8170--resnet_model-bn2b_branch2b-FusedBatchNormV2_grad-FusedBatchNormGradV2-0--float16.grad.bin", dtype=np.float16)
    dx_dumped = np.fromfile("11--8196--resnet_model-res2b_branch2b-Conv2D_grad-Conv2DBackpropInput-0--float16.grad.bin", dtype=np.float16)
    w  = np.fromfile("11--405--resnet_model-res2b_branch2b-kernel_cast-0--float16.var_cast.bin", dtype=np.float16)

    dx_dumped = dx_dumped.reshape(1,64,56,56)
    dy = dy.reshape(1,64,56,56)

    w  = w.reshape(3,3,64,64)

    stride = [1, 1, 1, 1]
    padding = 'SAME'
    dilation = [1, 1, 1, 1]

    grad_true = tf.nn.conv2d_backprop_input([1,64,56,56], w, dy, stride, padding, data_format="NCHW")

    result_grad = None
    with tf.Session():
        result_grad = grad_true.eval()

        result_grad.astype(np.float16).tofile("test_result_grad_fp16.bin")

        z = dx_dumped - result_grad
        n = tf.count_nonzero(z).eval()
        print("========")
        if n == 0:
            print("PASS: data just generated is the same as reference data")
        else:
            print("FAIL: data just generated is different than reference data, different count is [" + n + "]")
        print("========")

if __name__ == '__main__':
    main()

