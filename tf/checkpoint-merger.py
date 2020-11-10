
import os
import sys

import tensorflow as tf


if len(sys.argv) <= 1:
    print('Usage: {} <ckpt-in-1> [ckpt-in-2]...'.format(sys.argv[0]))
    exit()

ckpt_out = "my_concated_ckpt"
ckpt_out_log = ckpt_out + ".log"


ckpt_in = []
for ckpt in sys.argv[1:]:
    # validate ckpt
    tf.train.load_checkpoint(ckpt)
    ckpt_in.append(ckpt)




all_var_name_list_import = []
all_var_info_todo_list = []

tf.reset_default_graph()
with tf.Session() as sess:
    for ckpt in ckpt_in:
        ckpt_var_name_list = []
        ckpt_var_info_list_import = []
        ckpt_global_step = None
        for vname, vshape in tf.train.list_variables(ckpt):
            ckpt_var_name_list.append(vname)
            if vname not in all_var_name_list_import:
                all_var_name_list_import.append(vname) 
                ckpt_var_info_list_import.append([vname, vshape])
                v = tf.train.load_variable(ckpt, vname)
                v = tf.Variable(v, vname)
            if vname == "global_step":
                ckpt_global_step = tf.train.load_variable(ckpt, vname)

        all_var_info_todo_list.append([ckpt, ckpt_global_step, ckpt_var_info_list_import, ckpt_var_name_list])
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().save(sess, ckpt_out, write_meta_graph=False, write_state=False)

log_info = []
log_sum_info = []
log_sum_info.append("====summary==== (" + ckpt_out + ": total_imported=" + str(len(all_var_name_list_import)) + ")")
for ckpt, step, vinfo_list, vname_list in all_var_info_todo_list:
    info = "(" + ckpt + ": global_step=" + str(step) + " num_imported=" + str(len(vinfo_list)) + " num_total=" + str(len(vname_list)) + " num_dropped=" + str(len(vname_list) - len(vinfo_list)) + ")"
    log_sum_info.append(info)
    log_info.append(info)
    for vname, vshape in vinfo_list:
        log_info.append(str(vshape) + "\t" + vname)
    log_info.append("")


with open(ckpt_out_log, 'w') as f:
    for s in log_info + log_sum_info:
        print(s)
        f.write(s + '\n')
