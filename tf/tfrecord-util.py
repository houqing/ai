#!/usr/bin/env python3

import sys
import os
import tensorflow as tf 
from tensorflow.python.lib.io import file_io

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='9'

f_pattern = sys.argv[1]
#f_pattern = 'cn-wiki-128-f1/AA_wiki_00.tfrecord'

#f_split_num = sys.argv[2]
f_split_num = 10000

tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()

# TODO read a bunch of data files
print('pattern:', f_pattern)
ds = tf.data.TFRecordDataset(tf.gfile.Glob(f_pattern))

num = sum(1 for b in ds)

print('num:', num)
exit()


if f_split_num and int(f_split_num) > 0:
    #ds_split = ds.batch(f_split_num).make_one_shot_iterator().get_next()
    ds_split = ds.make_one_shot_iterator().get_next()
    part_id = 0
    for part in ds_split:
        f_out = f_pattern + '.{:06d}'.format(part_id)
        with tf.python_io.TFRecordWriter(f_out) as writer:
            #for r in part:
            #    writer.write(r)
            writer.write(part.numpy())
        part_id += 1
    # TODO do split

