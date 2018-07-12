#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import os

import numpy as np
import tensorflow as tf


fa="./data_a"
fb="./data_b"

fa_len = os.stat(fa).st_size
fb_len = os.stat(fb).st_size

if fa_len != fb_len:
    print("Warning: file length mismatch ", fa_len, "!=", fb_len)

data_len = min(fa_len, fb_len)
data_len = 32

fa_list = [fa]
fb_list = [fb]

fa_queue = tf.train.string_input_producer(fa_list)
fb_queue = tf.train.string_input_producer(fb_list)
reader_a = tf.FixedLengthRecordReader(record_bytes=data_len)
reader_b = tf.FixedLengthRecordReader(record_bytes=data_len)

key, value_a = reader_a.read(fa_queue)
key, value_b = reader_b.read(fb_queue)

values_a = tf.decode_raw(value_a, tf.float16)
values_b = tf.decode_raw(value_b, tf.float16)

values_a = tf.Print(values_a, [values_a], "values_a: ")
values_b = tf.Print(values_b, [values_b], "values_b: ")


sess = tf.Session()
tf.train.start_queue_runners(sess=sess)

a,b = sess.run([values_a, values_b])
print("a:b fp16")
print([i for i in a[:8]])
print([i for i in b[:8]])
print("a:b hex")
print([i for i in map(binascii.hexlify, a)])
print([i for i in map(binascii.hexlify, b)])

