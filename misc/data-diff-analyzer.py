#!/usr/bin/env python3

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys

import numpy as np
import matplotlib.pyplot as plt

def usage_exit(err_info='', err_no=-1):
    if err_info:
        print('Error: ', err_info)
    print('Usage:', sys.argv[0], '<file_a> <file_b> <f16|f32|npy>-<f16|f32|npy>')
    exit(err_no)

# check param
if len(sys.argv) not in [ 4, 5 ]:
    usage_exit()
is_need_cast_input_to_fp16 = False
is_skip_log = False
is_skip_fig = False
if len(sys.argv) == 5:
    if sys.argv[4] in [ 'f16', 'fp16', '16', 'h' ]:
        is_need_cast_input_to_fp16 = True
    elif sys.argv[4] in [ 'x' ]:
        is_skip_fig = True
    elif sys.argv[4] in [ 'a' ]:
        is_skip_log = True
        is_skip_log = True
    

# get input
f_a = sys.argv[1]
f_b = sys.argv[2]
f_out_log = f_b + '--diff.log'
f_out_pic = f_b + '--diff.png'
_f_type = sys.argv[3]
f_type = _f_type.split('-')

f_a_is_possibly_pad=True
if f_type[0] in [ 'f16', 'fp16', '16', 'h' ]:
    a = np.fromfile(f_a, np.float16)
    a_t = 'rfloat16'
elif f_type[0] in [ 'f32', 'fp32', '32', 's' ]:
    a = np.fromfile(f_a, np.float32)
    a_t = 'rfloat32'
elif f_type[0] in [ 'npy', 'np', 'n' ]:
    a = np.load(f_a).reshape(-1)
    f_a_is_possibly_pad=False
    a_t = 'n'+str(a.dtype)
else:
    usage_exit()

f_b_is_possibly_pad=True
if f_type[1] in [ 'f16', 'fp16', '16', 'h' ]:
    b = np.fromfile(f_b, np.float16)
    b_t = 'rfloat16'
elif f_type[1] in [ 'f32', 'fp32', '32', 's' ]:
    b = np.fromfile(f_b, np.float32)
    b_t = 'rfloat32'
elif f_type[1] in [ 'npy', 'np', 'n' ]:
    b = np.load(f_b).reshape(-1)
    f_b_is_possibly_pad=False
    b_t = 'n'+str(b.dtype)
else:
    usage_exit()

if is_need_cast_input_to_fp16:
    a = a.astype(np.float16)
    b = b.astype(np.float16)

a = a.astype(np.float64)
b = b.astype(np.float64)

# select input
#a = a[0:1000]
#b = b[0:1000]

# calc stat
_st_a_total = len(a)
_st_a_inf = len(np.argwhere(np.isinf(a)))
_st_a_nan = len(np.argwhere(np.isnan(a)))
_st_a_zero = len(np.argwhere(np.equal(a, 0)))
_st_b_total = len(b)
_st_b_inf = len(np.argwhere(np.isinf(b)))
_st_b_nan = len(np.argwhere(np.isnan(b)))
_st_b_zero = len(np.argwhere(np.equal(b, 0)))
st_a_info = 'total='+str(_st_a_total)+' inf='+str(_st_a_inf)+' nan='+str(_st_a_nan)+' zero='+str(_st_a_zero)
st_b_info = 'total='+str(_st_b_total)+' inf='+str(_st_b_inf)+' nan='+str(_st_b_nan)+' zero='+str(_st_b_zero)

# generate output info head
f_a_info = 'A_'+a_t+' : '+f_a
f_b_info = 'B_'+b_t+' : '+f_b

output_info_head = []
output_info_head.append(f_a_info)
output_info_head.append(f_b_info)
if False:
    output_info_head.append('log   : ' + f_out_log)
    output_info_head.append('pic   : ' + f_out_pic)
output_info_head.append('info_a: ' + st_a_info)
output_info_head.append('info_b: ' + st_b_info)

# print output info head
for i in output_info_head:
    print(i)

# check error
if _st_a_total != _st_b_total:
    if ((_st_a_total + _st_b_total) > 64):
        if (_st_a_total < _st_b_total) and ((_st_a_total * 2) < _st_b_total):
            print('Error: Possibly incorrect data type for A<B')
            exit(1)
        if (_st_a_total > _st_b_total) and (_st_a_total > (_st_b_total * 2)):
            print('Error: Possibly incorrect data type for A>B')
            exit(1)
    if f_b_is_possibly_pad and (_st_b_total > _st_a_total):
        b = b[0:len(a)]
    elif f_a_is_possibly_pad and (_st_a_total > _st_b_total):
        a = a[0:len(a)]
    else:
        print('Error: A, B size mismatch')
        exit(1)

# process input
_arg_sort = np.argsort(a)
aa = np.take(a, _arg_sort)
bb = np.take(b, _arg_sort)

# select from sorted
if False:
    _sel_begin = 65127
    _sel_end = 65128
    _sel_begin = 65120
    _sel_end = 65140
    aa = aa[_sel_begin:_sel_end]
    bb = bb[_sel_begin:_sel_end]
    #aa = np.array([0x7FFFFFFFFFFFFFFF], np.uint64).view(np.float64)
    #bb = np.array([0xf], np.uint64).view(np.float64)
    print("aa:", aa)
    print("bb:", bb)

# function to generate data averages
def gen_avg_all(data):
    data_avg_s = np.mean(data) if len(data) else 0
    data_avg = np.array(data)
    data_avg.fill(data_avg_s)
    return data_avg, data_avg_s

def gen_avg_pos_neg(data):
    _arg_data_pos = np.argwhere(np.greater(data, 0))
    _data_pos = np.take(data, _arg_data_pos)
    data_pos_avg = np.array(data)
    data_pos_avg_s = np.mean(_data_pos) if len(_data_pos) else 0
    data_pos_avg.fill(data_pos_avg_s)

    _arg_data_neg = np.argwhere(np.less(data, 0))
    _data_neg = np.take(data, _arg_data_neg)
    data_neg_avg_s = np.mean(_data_neg) if len(_data_neg) else 0
    data_neg_avg = np.array(data)
    data_neg_avg.fill(data_neg_avg_s)

    return data_pos_avg, data_pos_avg_s, data_neg_avg, data_neg_avg_s


# calc data, avgs
aa_pos_avg, aa_pos_avg_s, aa_neg_avg, aa_neg_avg_s = gen_avg_pos_neg(aa)
bb_pos_avg, bb_pos_avg_s, bb_neg_avg, bb_neg_avg_s = gen_avg_pos_neg(bb)

# calc diff, avgs
diff = bb - aa
diff_abs = abs(diff)
diff_pos_avg, diff_pos_avg_s, diff_neg_avg, diff_neg_avg_s = gen_avg_pos_neg(diff)
diff_abs_avg, diff_abs_avg_s = gen_avg_all(diff_abs)

# calc rel diff, avg
diff_denom = np.maximum(abs(aa), abs(bb))

_sum_abs = abs(aa) + abs(bb)
_arg_zeros = np.argwhere(np.equal(_sum_abs, 0))
np.put(_sum_abs, _arg_zeros, 1)
diff_rel = abs(aa - bb) / _sum_abs
np.put(diff_rel, _arg_zeros, 0)
diff_rel_avg, diff_rel_avg_s = gen_avg_all(diff_rel)


# calc ideal diff, avg
_A_u64 = aa.view(np.int64)
_A_f64 = _A_u64.astype(np.float64)
_B_u64 = bb.view(np.int64)
_B_f64 = _B_u64.astype(np.float64)

_sub_abs = abs(_A_f64 - _B_f64)
_sum_abs = abs(_A_f64) + abs(_B_f64)

_arg_zeros = np.argwhere(np.equal(_sum_abs, 0))
np.put(_sum_abs, _arg_zeros, 1)
diff_rel_ideal = _sub_abs / _sum_abs
np.put(diff_rel_ideal, _arg_zeros, 0)

diff_rel_ideal_avg, diff_rel_ideal_avg_s = gen_avg_all(diff_rel_ideal)

# generate output info tail
data_a_info = 'avg_pos='+str(aa_pos_avg_s)+' avg_neg='+str(aa_neg_avg_s)
data_b_info = 'avg_pos='+str(bb_pos_avg_s)+' avg_neg='+str(bb_neg_avg_s)
diff_info = 'avg_pos='+str(diff_pos_avg_s)+' avg_neg='+str(diff_neg_avg_s)+' avg_abs='+str(diff_abs_avg_s)
diff_rel_info = 'avg_rel='+str(diff_rel_avg_s)
diff_rel_ideal_info = 'avg_idl='+str(diff_rel_ideal_avg_s)

output_info_tail = []
output_info_tail.append('data_a: ' + data_a_info)
output_info_tail.append('data_b: ' + data_b_info)
output_info_tail.append('diff_abs: ' + diff_info)
output_info_tail.append('diff_rel: ' + diff_rel_info)
output_info_tail.append('diff_idl: ' + diff_rel_ideal_info)

# print output info tail
for i in output_info_tail:
    print(i)

# output text
if not is_skip_log:
    with open(f_out_log, 'w') as f:
        for i in output_info_head:
            f.write(i + '\n')
        for i in output_info_tail:
            f.write(i + '\n')

# check if need figure
if not is_skip_fig:
    # output figure
    fig_title = f_a_info + '\n' + f_b_info
    fig_avg_linewidth = 0.5
    fig_diff_abs_linewidth = 0.5
    fig_markersize_a = 1
    fig_markersize_b = 0.5
    fig_markersize = 1

    plt.figure(1, figsize=(24, 12))

    ax1 = plt.subplot(411)
    plt.title(fig_title, loc='left')
    plt.plot(aa, label='data a ('+st_a_info+' '+data_a_info+')', linewidth=0, marker='.', markersize=fig_markersize_a, markeredgewidth=0, markerfacecolor='red')
    plt.plot(aa_pos_avg, linewidth=fig_avg_linewidth, color='red')
    plt.plot(aa_neg_avg, linewidth=fig_avg_linewidth, color='green')
    plt.plot(bb, label='data b ('+st_b_info+' '+data_b_info+')', linewidth=0, marker='.', markersize=fig_markersize_b, markeredgewidth=0, markerfacecolor='green')
    plt.legend(loc='upper left')
    plt.ylabel('(data value)')

    ax2 = plt.subplot(412, sharex=ax1)
    ax2.xaxis.set_visible(False)
    plt.plot(diff, label='diff [b-a]  ('+diff_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='black')
    plt.plot(diff_pos_avg, linewidth=fig_avg_linewidth, color='red')
    plt.plot(diff_neg_avg, linewidth=fig_avg_linewidth, color='green')
    #plt.plot(diff_abs_avg, linewidth=fig_diff_abs_linewidth, color='black')
    plt.legend(loc='upper left')

    ax3 = plt.subplot(413, sharex=ax1)
    ax3.xaxis.set_visible(False)
    plt.plot(diff_rel, label='diff [|a-b|/(|a|+|b|)]  ('+diff_rel_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='black')
    plt.plot(diff_rel_avg, linewidth=fig_avg_linewidth, color='red')
    plt.legend(loc='upper left')

    ax4 = plt.subplot(414, sharex=ax1)
    ax4.xaxis.set_visible(False)
    plt.plot(diff_rel_ideal, label='diff ideal ('+diff_rel_ideal_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='black')
    plt.plot(diff_rel_ideal_avg, linewidth=fig_avg_linewidth, color='red')
    plt.legend(loc='upper left')

    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.01)
    plt.subplots_adjust(hspace=0.01)
    plt.savefig(f_out_pic, bbox_inches='tight')

