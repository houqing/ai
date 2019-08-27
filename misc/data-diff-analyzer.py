#!/usr/bin/env python3

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys

import numpy as np
import matplotlib.pyplot as plt

VER=8

def usage_exit(err_info='', err_no=-1):
    if err_info:
        print('Error: ', err_info)
    print('Usage:', sys.argv[0], '<file_a> <file_b> <f16|f32|npy>-<f16|f32|npy>')
    exit(err_no)

# check param
if len(sys.argv) < 4:
    usage_exit()
is_need_cast_input_to_fp16 = False
is_skip_log = False
is_skip_fig = False
is_marker_factor_auto = False
is_sort = True
if len(sys.argv) >= 5:
    for arg in sys.argv[4:]:
        if arg in [ 'f16', 'fp16', '16', 'h' ]:
            is_need_cast_input_to_fp16 = True
        elif arg in [ 'nofig' ]:
            is_skip_fig = True
        elif arg in [ 'nolog' ]:
            is_skip_log = True
        elif arg in [ 'auto' ]:
            is_marker_factor_auto = True
        elif arg in [ 'nosort' ]:
            is_sort = False
    
# get input
f_a = sys.argv[1]
f_b = sys.argv[2]
f_out_log = f_b + '--diff.log'
f_out_pic = f_b + '--diff.png'
_f_type = sys.argv[3]
f_type = _f_type.split('-')

f_a_is_possibly_pad=True
a_dtype = np.float32
if f_type[0] in [ 'f16', 'fp16', '16', 'h' ]:
    a = np.fromfile(f_a, np.float16)
    a_t = 'rfloat16'
    a_dtype = np.float16
elif f_type[0] in [ 'f32', 'fp32', '32', 's' ]:
    a = np.fromfile(f_a, np.float32)
    a_t = 'rfloat32'
    a_dtype = np.float32
elif f_type[0] in [ 'npy', 'np', 'n' ]:
    a = np.load(f_a).reshape(-1)
    f_a_is_possibly_pad=False
    a_t = 'n'+str(a.dtype)
    if a.dtype == np.float32:
        a_dtype = np.float32
    elif a.dtype == np.float16:
        a_dtype = np.float16
    else:
        usage_exit('not support npy dtype for a: '+str(a.dtype))
else:
    usage_exit()

f_b_is_possibly_pad=True
b_dtype = np.float32
if f_type[1] in [ 'f16', 'fp16', '16', 'h' ]:
    b = np.fromfile(f_b, np.float16)
    b_t = 'rfloat16'
    b_dtype = np.float16
elif f_type[1] in [ 'f32', 'fp32', '32', 's' ]:
    b = np.fromfile(f_b, np.float32)
    b_t = 'rfloat32'
    b_dtype = np.float32
elif f_type[1] in [ 'npy', 'np', 'n' ]:
    b = np.load(f_b).reshape(-1)
    f_b_is_possibly_pad=False
    b_t = 'n'+str(b.dtype)
    if b.dtype == np.float32:
        b_dtype = np.float32
    elif b.dtype == np.float16:
        b_dtype = np.float16
    else:
        usage_exit('not support npy dtype for b: '+str(b.dtype))
else:
    usage_exit()

ab_min_dtype = np.float32
if a_dtype == np.float16 or b_dtype == np.float16:
    ab_min_dtype = np.float16

if is_need_cast_input_to_fp16:
    a = a.astype(np.float16)
    b = b.astype(np.float16)

a = a.astype(np.float64)
b = b.astype(np.float64)

# select input
st_a_total_orig = len(a)
st_b_total_orig = len(b)
st_real_data_total = st_a_total_orig
is_trimmed = False
np.random.seed(0x1234)
if len(f_type) == 3:
    st_real_data_total = min(st_a_total_orig, st_b_total_orig)
    _arg_pick = np.random.permutation(st_real_data_total)

    st_real_data_total = min(st_real_data_total, int(f_type[2]))
    st_real_data_total = max(st_real_data_total, 1)
    _arg_pick = _arg_pick[:st_real_data_total]
    if is_sort:
        _arg_pick = _arg_pick
    else:
        _arg_pick.sort()

    a = np.take(a, _arg_pick)
    b = np.take(b, _arg_pick)
    is_trimmed = True

# calc stat
st_a_total = len(a)
_st_a_inf = len(np.argwhere(np.isinf(a)))
_st_a_nan = len(np.argwhere(np.isnan(a)))
_st_a_zero = len(np.argwhere(np.equal(a, 0)))
st_b_total = len(b)
_st_b_inf = len(np.argwhere(np.isinf(b)))
_st_b_nan = len(np.argwhere(np.isnan(b)))
_st_b_zero = len(np.argwhere(np.equal(b, 0)))
_st_a_trim_info = '<'+str(st_a_total_orig)+'>' if is_trimmed else ''
_st_b_trim_info = '<'+str(st_b_total_orig)+'>' if is_trimmed else ''
st_a_info = 'total='+str(st_a_total)+_st_a_trim_info+' inf='+str(_st_a_inf)+' nan='+str(_st_a_nan)+' zero='+str(_st_a_zero)
st_b_info = 'total='+str(st_b_total)+_st_b_trim_info+' inf='+str(_st_b_inf)+' nan='+str(_st_b_nan)+' zero='+str(_st_b_zero)

# generate output info head
f_a_info = 'A_'+a_t+'--v'+str(VER)+': '+f_a
f_b_info = 'B_'+b_t+'--v'+str(VER)+': '+f_b

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
if st_a_total != st_b_total:
    if ((st_a_total + st_b_total) > 64):
        if (st_a_total < st_b_total) and ((st_a_total * 2) < st_b_total):
            print('Error: Possibly incorrect data type for A<B')
            exit(1)
        if (st_a_total > st_b_total) and (st_a_total > (st_b_total * 2)):
            print('Error: Possibly incorrect data type for A>B')
            exit(1)
    _is_len_corrected = False
    if f_b_is_possibly_pad or f_a_is_possibly_pad:
        if f_b_is_possibly_pad and (st_b_total > st_a_total):
            b = b[0:st_a_total]
            _is_len_corrected = True
        if f_a_is_possibly_pad and (st_a_total > st_b_total):
            a = a[0:st_b_total]
            _is_len_corrected = True
    if not _is_len_corrected:
        print('Error: A, B size mismatch')
        exit(1)

# process input
if is_sort:
    _arg_sort = np.argsort(a)
    aa = np.take(a, _arg_sort)
    bb = np.take(b, _arg_sort)
else:
    aa = a
    bb = b

# select from sorted
if True:
    _sel_begin = 65127
    _sel_end = 65128
    _sel_begin = 5200
    _sel_end = 5400
    _sel_begin = 3800
    _sel_end = 6600
    _sel_begin = 4500
    _sel_end = 5400
    aa = aa[_sel_begin:_sel_end]
    bb = bb[_sel_begin:_sel_end]
    #aa = np.array([0x3FEFFF00], np.uint64).view(np.float64)
    #bb = np.array([0x3FFF0], np.uint64).view(np.float64)
    print("aa:", aa[:10])
    print("bb:", bb[:10])

# function to generate data averages
def gen_avg_all(data):
    data_avg_s = np.mean(data) if len(data) else 0
    return data_avg_s

def gen_avg_pos_neg(data):
    _arg_data_pos = np.argwhere(np.greater(data, 0))
    _data_pos = np.take(data, _arg_data_pos)
    data_pos_avg_s = np.mean(_data_pos) if len(_data_pos) else 0

    _arg_data_neg = np.argwhere(np.less(data, 0))
    _data_neg = np.take(data, _arg_data_neg)
    data_neg_avg_s = np.mean(_data_neg) if len(_data_neg) else 0

    return data_pos_avg_s, data_neg_avg_s


# calc data, avgs
aa_pos_avg_s, aa_neg_avg_s = gen_avg_pos_neg(aa)
bb_pos_avg_s, bb_neg_avg_s = gen_avg_pos_neg(bb)

# calc abs diff, avgs
diff_abs = bb - aa
diff_abs_pos_avg_s, diff_abs_neg_avg_s = gen_avg_pos_neg(diff_abs)

# calc rel diff, avg
_sum_abs = abs(aa) + abs(bb)
_sub_abs = abs(aa - bb)
_arg_zeros = np.argwhere(np.equal(_sum_abs, 0))
np.put(_sum_abs, _arg_zeros, 1)
diff_rel = _sub_abs / _sum_abs
np.put(diff_rel, _arg_zeros, 0)
diff_rel_avg_s = gen_avg_all(diff_rel)

_diff_avg_mod = diff_rel_avg_s * 100
_arg_norm = np.argwhere(np.less_equal(diff_rel, _diff_avg_mod))
_arg_mod = np.argwhere(np.greater(diff_rel, _diff_avg_mod))
diff_rel_norm = np.array(diff_rel)
diff_rel_mod = np.array(diff_rel)
np.put(diff_rel_norm, _arg_mod, 0)
np.put(diff_rel_mod, _arg_mod, _diff_avg_mod)
np.put(diff_rel_mod, _arg_norm, 0)

# calc ideal diff, avg
_A_s64 = aa.view(np.int64)
_A_u64 = aa.view(np.uint64)
_A_u64_abs = _A_u64 & 0x7fffffffffffffff
_B_s64 = bb.view(np.int64)
_B_u64 = bb.view(np.uint64)
_B_u64_abs = _B_u64 & 0x7fffffffffffffff

_A_u64_sign = (_A_u64 & 0x8000000000000000) >> 63
_B_u64_sign = (_B_u64 & 0x8000000000000000) >> 63
_AB_u64_sign_is_different = np.logical_xor(_A_u64_sign, _B_u64_sign)
_AB_u64_sign_is_same = np.logical_not(_AB_u64_sign_is_different)

# for f16: 0x1:6e-8:0x3E701B2B29A4692B 0x3:2e-7:0x3E8AD7F29ABCAF48 0x7:4e-7:0x3E9AD7F29ABCAF48 0x1f:1.85e-6:0x3EBF09B082EA2AAC 0x3f:3.76e-6:0x3ECF8A89DC374DF5
_ref_u64_min_for_f16 = 0x3E9AD7F29ABCAF48
# for f32: 0x1:1e-45:0x3696D601AD376AB9 0x3:4e-45:0x36B6D601AD376AB9 0x7:1e-44:0x36CC8B8218854567 0x1f:4.3e-44:0x36EEAF9240C27769
_ref_u64_min_for_f32 = 0x36CC8B8218854567
if ab_min_dtype == np.float32:
    _ref_u64_min = _ref_u64_min_for_f32
elif ab_min_dtype == np.float16:
    _ref_u64_min = _ref_u64_min_for_f16
_AB_u64_sign_is_same = np.where(_A_u64_abs < _ref_u64_min, True, _AB_u64_sign_is_same)
_B_u64_abs = np.where(_A_u64_abs < _ref_u64_min, _B_u64_abs + _ref_u64_min, _B_u64_abs)
_A_u64_abs = np.where(_A_u64_abs < _ref_u64_min, _ref_u64_min, _A_u64_abs)

_AB_u64_sign_is_same = np.where(_B_u64_abs < _ref_u64_min, True, _AB_u64_sign_is_same)
_A_u64_abs = np.where(_B_u64_abs < _ref_u64_min, _A_u64_abs + _ref_u64_min, _A_u64_abs)
_B_u64_abs = np.where(_B_u64_abs < _ref_u64_min, _ref_u64_min, _B_u64_abs)

print('A:', _A_u64_abs[:10])
print('B:', _B_u64_abs[:10])
print('AB_sign:', _AB_u64_sign_is_same[:10])
_sum_abs = _A_u64_abs + _B_u64_abs
_sub_AB_abs = _A_u64_abs - _B_u64_abs
_sub_BA_abs = _B_u64_abs - _A_u64_abs
print('sum:', _sum_abs[:10])
_sub_abs = _sum_abs
_sub_abs = np.where(np.logical_and(_AB_u64_sign_is_same, _A_u64_abs >= _B_u64_abs), _sub_AB_abs, _sub_abs)
_sub_abs = np.where(np.logical_and(_AB_u64_sign_is_same, _A_u64_abs < _B_u64_abs), _sub_BA_abs, _sub_abs)
print('sub:', _sub_abs[:10])
_arg_zeros = np.argwhere(np.equal(_sum_abs, 0))
np.put(_sum_abs, _arg_zeros, 1)
diff_rel_ideal = _sub_abs / _sum_abs
np.put(diff_rel_ideal, _arg_zeros, 0)
diff_rel_ideal_avg_s = gen_avg_all(diff_rel_ideal)

_diff_avg_mod = diff_rel_ideal_avg_s * 100
_arg_norm = np.argwhere(np.less_equal(diff_rel_ideal, _diff_avg_mod))
_arg_mod = np.argwhere(np.greater(diff_rel_ideal, _diff_avg_mod))
diff_rel_ideal_norm = np.array(diff_rel_ideal)
diff_rel_ideal_mod = np.array(diff_rel_ideal)
np.put(diff_rel_ideal_norm, _arg_mod, 0)
np.put(diff_rel_ideal_mod, _arg_mod, _diff_avg_mod)
np.put(diff_rel_ideal_mod, _arg_norm, 0)

if False:
    _A_s64 = aa.view(np.int64)
    _A_u64 = aa.view(np.uint64)
    _A_f64 = _A_s64.astype(np.float64)
    _B_s64 = bb.view(np.int64)
    _B_u64 = bb.view(np.uint64)
    _B_f64 = _B_s64.astype(np.float64)
    _sub_abs = abs(_A_f64 - _B_f64)
    _sum_abs = abs(_A_f64) + abs(_B_f64)
    _sum_u64_abs = abs(_A_u64 & 0x7fffffffffffffff) + abs(_B_u64 & 0x7fffffffffffffff)
    _arg_zeros = np.argwhere(np.equal(_sum_u64_abs, 0))
    np.put(_sum_abs, _arg_zeros, 1)
    diff_rel_ideal = _sub_abs / _sum_abs
    np.put(diff_rel_ideal, _arg_zeros, 0)
    diff_rel_ideal_avg_s = gen_avg_all(diff_rel_ideal)

    _diff_avg_mod = diff_rel_ideal_avg_s * 100
    _arg_norm = np.argwhere(np.less_equal(diff_rel_ideal, _diff_avg_mod))
    _arg_mod = np.argwhere(np.greater(diff_rel_ideal, _diff_avg_mod))
    diff_rel_ideal_norm = np.array(diff_rel_ideal)
    diff_rel_ideal_mod = np.array(diff_rel_ideal)
    np.put(diff_rel_ideal_norm, _arg_mod, 0)
    np.put(diff_rel_ideal_mod, _arg_mod, _diff_avg_mod)
    np.put(diff_rel_ideal_mod, _arg_norm, 0)

# generate output info tail
data_a_info = 'avg_pos='+str(aa_pos_avg_s)+' avg_neg='+str(aa_neg_avg_s)
data_b_info = 'avg_pos='+str(bb_pos_avg_s)+' avg_neg='+str(bb_neg_avg_s)
diff_info = 'avg_pos='+str(diff_abs_pos_avg_s)+' avg_neg='+str(diff_abs_neg_avg_s)
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
    fig_alpha = 0.5
    if is_marker_factor_auto:
        _fig_markersize_factor = max(0.11, (30 / np.log(st_real_data_total + 1) + -1.55))
    else:
        _fig_markersize_factor = 1
    fig_markersize_a = 1.6 * _fig_markersize_factor
    fig_markersize_b = 1.5 * _fig_markersize_factor
    fig_markersize_mod = 1.5 * _fig_markersize_factor
    fig_markersize = 1.5 * _fig_markersize_factor
    fig_legend_fontsize = 'x-small'
    _is_draw_avg_line = False

    plt.figure(1, figsize=(20, 10))

    ax1 = plt.subplot(411, xbound=110)
    ax1.xaxis.tick_top()
    ax1.axhline(aa_pos_avg_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(aa_neg_avg_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth)
    plt.title(fig_title, loc='left')
    plt.plot(aa, label='data a ('+st_a_info+') ('+data_a_info+')', linewidth=0, marker='.', markersize=fig_markersize_a, markeredgewidth=0, markerfacecolor='red', alpha=fig_alpha)
    plt.plot(bb, label='data b ('+st_b_info+') ('+data_b_info+')', linewidth=0, marker='.', markersize=fig_markersize_b, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax2 = plt.subplot(412)
    ax2.xaxis.set_visible(False)
    ax2.axhline(diff_abs_pos_avg_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth)
    ax2.axhline(diff_abs_neg_avg_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth)
    plt.plot(diff_abs, label='diff [b-a]  ('+diff_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax3 = plt.subplot(413)
    ax3.xaxis.set_visible(False)
    ax3.axhline(diff_rel_avg_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth)
    plt.plot(diff_rel_mod, linewidth=0, marker='.', markersize=fig_markersize_mod, markeredgewidth=0, markerfacecolor='red', alpha=fig_alpha)
    plt.plot(diff_rel_norm, label='diff [|a-b|/(|a|+|b|)]  ('+diff_rel_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax4 = plt.subplot(414)
    ax4.xaxis.set_visible(False)
    ax4.axhline(diff_rel_ideal_avg_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth)
    plt.plot(diff_rel_ideal_mod, linewidth=0, marker='.', markersize=fig_markersize_mod, markeredgewidth=0, markerfacecolor='red', alpha=fig_alpha)
    plt.plot(diff_rel_ideal_norm, label='diff ideal ('+diff_rel_ideal_info+')', linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f_out_pic, bbox_inches='tight')

