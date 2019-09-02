#!/usr/bin/env python3

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys

import numpy as np
import matplotlib.pyplot as plt

VER=9

def usage_exit(err_info='', err_no=-1):
    if err_info:
        print('Error: ', err_info)
    print('Usage:', sys.argv[0], '<file_a> <file_b> <fp16|fp32|npy>-<fp16|fp32|npy>')
    exit(err_no)

# check param
if len(sys.argv) < 4:
    usage_exit()
is_need_cast_input_to_fp16 = False
is_ideal_need_align_dtype = False
is_skip_log = False
is_skip_fig = False
is_marker_factor_auto = False
is_sort = False
conf_trim_off = None
conf_trim_len = None
conf_fig_format = 'png'
if len(sys.argv) >= 5:
    for arg in sys.argv[4:]:
        if arg in [ 'sort' ]:
            is_sort = True
        elif arg in [ 'nofig', 'nopic' ]:
            is_skip_fig = True
        elif arg in [ 'nolog' ]:
            is_skip_log = True
        elif arg in [ 'nofile' ]:
            is_skip_fig = True
            is_skip_log = True
        # XXX debug only options, bugs are features here
        elif arg in [ 'jpg', 'jpeg', 'svg', 'pdf' ]:
            conf_fig_format = arg
        elif arg in [ 'auto' ]:
            is_marker_factor_auto = True
        elif arg in [ 'f16', 'fp16', '16', 'h' ]:
            is_need_cast_input_to_fp16 = True
        elif arg in [ 'ideal-align-dtype', 'align-dtype', 'align' ]:
            is_ideal_need_align_dtype = True
        elif arg.startswith('offset=') or arg.startswith('off='):
            conf_trim_off = int(arg.lstrip('offset='))
        elif arg.startswith('length=') or arg.startswith('len='):
            conf_trim_len = int(arg.lstrip('length='))
        else:
            usage_exit('unknown parameter "'+arg+'"')

# init
np.random.seed(0x1234)
    
# get input
f_a = sys.argv[1]
f_b = sys.argv[2]
f_out_log = f_b + '--sort-diff.log' if is_sort else f_b + '--diff.log'
f_out_fig = f_b + '--sort-diff.' + conf_fig_format if is_sort else f_b + '--diff.' + conf_fig_format
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
ab_max_dtype = np.float16
if a_dtype == np.float16 or b_dtype == np.float16:
    ab_min_dtype = np.float16
if a_dtype == np.float32 or b_dtype == np.float32:
    ab_max_dtype = np.float32

conf_ideal_type = np.float32
if is_ideal_need_align_dtype:
    if ab_max_dtype == np.float16 or is_need_cast_input_to_fp16:
        conf_ideal_type = np.float16
        is_need_cast_input_to_fp16 = True

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
if True:
    output_info_head.append('log   : ' + f_out_log)
    output_info_head.append('fig   : ' + f_out_fig)
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

# select from list
_sel_begin = 0
_sel_end = st_real_data_total
if conf_trim_off and conf_trim_off > 0:
    if conf_trim_off < st_real_data_total:
        _sel_begin = conf_trim_off
if conf_trim_len and conf_trim_len > 0:
    _sel_end = _sel_begin + conf_trim_len
    if _sel_end > st_real_data_total:
        _sel_end = st_real_data_total
    st_real_data_total = _sel_end - _sel_begin
aa = aa[_sel_begin:_sel_end]
bb = bb[_sel_begin:_sel_end]
#aa = np.array([0x3FEFFF00], np.uint64).view(np.float64)
#bb = np.array([0x3FFF0], np.uint64).view(np.float64)
#print("aa:", aa[:10])
#print("bb:", bb[:10])

# function to generate data averages
def gen_avg_all(data):
    data_avg_s = np.mean(data) if len(data) else 0
    return data_avg_s

def gen_max_min_all(data):
    data_max_s = np.nanmax(data) if len(data) else 0
    data_min_s = np.nanmin(data) if len(data) else 0
    return data_max_s, data_min_s

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
aa_max_s, aa_min_s = gen_max_min_all(aa)
bb_pos_avg_s, bb_neg_avg_s = gen_avg_pos_neg(bb)
bb_max_s, bb_min_s = gen_max_min_all(bb)
ab_max_s = max(aa_max_s, bb_max_s)
ab_min_s = min(aa_min_s, bb_min_s)

# calc abs diff, avgs
diff_inc = bb - aa
diff_inc_pos_avg_s, diff_inc_neg_avg_s = gen_avg_pos_neg(diff_inc)
diff_inc_max_s, diff_inc_min_s = gen_max_min_all(diff_inc)
_arg_non_zeros = np.argwhere(np.not_equal(diff_inc, 0.0))
diff_inc_diff_num = len(_arg_non_zeros)

# calc rel diff, avg
_sum_abs = abs(aa) + abs(bb)
_sub_abs = abs(aa - bb)
_arg_zeros = np.argwhere(np.equal(_sum_abs, 0))
np.put(_sum_abs, _arg_zeros, 1)
diff_rel = _sub_abs / _sum_abs
np.put(diff_rel, _arg_zeros, 0)
diff_rel_avg_s = gen_avg_all(diff_rel)
diff_rel_max_s, diff_rel_min_s = gen_max_min_all(diff_rel)

# calc ideal diff, avg
def gen_ideal_diff(dtype=np.float32):
    _is_enable_norm = False
    if dtype in [ np.float32 ]:
        _AB_dtype_f = np.float32
        _AB_dtype_u = np.uint32
        _AB_u_abs_mask = 0x7fffffff
        _AB_u_sign_mask = 0x80000000
        _AB_u_sign_shift = 31
        _diff_ideal_mark_s = np.log2(np.array(0x1, np.uint16).view(np.float16).astype(np.float32).view(np.uint32))
        _diff_ideal_man_mark_s = np.log2(np.array(0x7fffff, np.uint32))
    elif dtype in [ np.float16 ]:
        _AB_dtype_f = np.float16
        _AB_dtype_u = np.uint16
        _AB_u_abs_mask = 0x7fff
        _AB_u_sign_mask = 0x80000000
        _AB_u_sign_shift = 15
        _diff_ideal_mark_s = np.log2(np.array(0x3ff, np.uint16).view(np.float16).astype(np.float16).view(np.uint16))
        _diff_ideal_man_mark_s = np.log2(np.array(0x3ff, np.uint16))

    A_f = aa.astype(_AB_dtype_f)
    A_u = A_f.view(_AB_dtype_u)
    A_u64_abs = (A_u & _AB_u_abs_mask).astype(np.uint64)
    A_u_sign = (A_u & _AB_u_sign_mask) >> _AB_u_sign_shift
    B_f = bb.astype(_AB_dtype_f)
    B_u = B_f.view(_AB_dtype_u)
    B_u64_abs = (B_u & _AB_u_abs_mask).astype(np.uint64)
    B_u_sign = (B_u & _AB_u_sign_mask) >> _AB_u_sign_shift

    _AB_u_sign_is_different = np.logical_xor(A_u_sign, B_u_sign)
    _AB_u_sign_is_same = np.logical_not(_AB_u_sign_is_different)

    # calc for different sign
    AB_diff_bits = np.log2(A_u64_abs + B_u64_abs + 1)   # default as different sign

    # calc for same sign
    _AB_u64_sub_fix = np.where(_AB_u_sign_is_same, np.maximum(A_u64_abs, B_u64_abs) - np.minimum(A_u64_abs, B_u64_abs) + 1, 1)
    AB_diff_bits = np.where(_AB_u_sign_is_same, np.log2(_AB_u64_sub_fix), AB_diff_bits)

    if _is_enable_norm:
        _diff_ideal_mark_s = _diff_ideal_mark_s / (_AB_u_sign_shift + 1)
        _diff_ideal = AB_diff_bits / (_AB_u_sign_shift + 1)
    else:
        _diff_ideal_mark_s = _diff_ideal_mark_s
        _diff_ideal = AB_diff_bits
    _diff_ideal_avg_s = gen_avg_all(_diff_ideal)

    return _diff_ideal, _diff_ideal_avg_s, _diff_ideal_man_mark_s, _diff_ideal_mark_s

diff_ideal_f32, diff_ideal_f32_avg_s, diff_ideal_f32_man_mark_s, diff_ideal_f32_mark_for_f16_s = gen_ideal_diff(np.float32)
diff_ideal_f32_max_s, diff_ideal_f32_min_s = gen_max_min_all(diff_ideal_f32)
diff_ideal_f16, diff_ideal_f16_avg_s, diff_ideal_f16_man_mark_s, diff_ideal_f16_mark_for_f16_s = gen_ideal_diff(np.float16)
diff_ideal_f16_max_s, diff_ideal_f16_min_s = gen_max_min_all(diff_ideal_f16)

# generate output info tail
data_a_info = 'avg_pos='+str(aa_pos_avg_s)+' avg_neg='+str(aa_neg_avg_s)+' max='+str(aa_max_s)+' min='+str(aa_min_s)
data_b_info = 'avg_pos='+str(bb_pos_avg_s)+' avg_neg='+str(bb_neg_avg_s)+' max='+str(bb_max_s)+' min='+str(bb_min_s)
diff_info = 'diff_num='+str(diff_inc_diff_num)+' avg_pos='+str(diff_inc_pos_avg_s)+' avg_neg='+str(diff_inc_neg_avg_s)+' max='+str(diff_inc_max_s)+' min='+str(diff_inc_min_s)
diff_rel_info = 'avg='+str(diff_rel_avg_s)+' max='+str(diff_rel_max_s)
diff_ideal_f32_info = 'avg='+str(diff_ideal_f32_avg_s)+' max='+str(diff_ideal_f32_max_s)+' min='+str(diff_ideal_f32_min_s)
diff_ideal_f16_info = 'avg='+str(diff_ideal_f16_avg_s)+' max='+str(diff_ideal_f16_max_s)+' min='+str(diff_ideal_f16_min_s)

output_info_tail = []
output_info_tail.append('data_a: ' + data_a_info)
output_info_tail.append('data_b: ' + data_b_info)
output_info_tail.append('diff_inc: ' + diff_info)
output_info_tail.append('diff_rel: ' + diff_rel_info)
output_info_tail.append('diff_idl_f32: ' + diff_ideal_f32_info)
output_info_tail.append('diff_idl_f16: ' + diff_ideal_f16_info)

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
    fig_thresh_linewidth = 0.05
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

    ax1 = plt.subplot(511, xbound=110)
    ax1.xaxis.tick_top()
    ax1.axhline(aa_pos_avg_s, xmax=0.005, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(aa_neg_avg_s, xmax=0.005, color='green', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(bb_pos_avg_s, xmax=0.01, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(bb_neg_avg_s, xmax=0.01, color='green', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(aa_max_s, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(aa_min_s, xmax=0.015, color='green', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(bb_max_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax1.axhline(bb_min_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth, marker=None)
    plt.title(fig_title, loc='left')
    plt.plot(aa, label='data_a: '+st_a_info+' '+data_a_info, linewidth=0, marker='.', markersize=fig_markersize_a, markeredgewidth=0, markerfacecolor='red', alpha=fig_alpha)
    plt.plot(bb, label='data_b: '+st_b_info+' '+data_b_info, linewidth=0, marker='.', markersize=fig_markersize_b, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax2 = plt.subplot(512)
    ax2.xaxis.set_visible(False)
    ax2.axhline(diff_inc_pos_avg_s, xmax=0.01, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax2.axhline(diff_inc_neg_avg_s, xmax=0.01, color='green', linewidth=fig_avg_linewidth, marker=None)
    if False:
        ax2.axhline(diff_inc_max_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax2.axhline(diff_inc_min_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth, marker=None)
    plt.plot(diff_inc, label='diff_inc (=b-a): '+diff_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax3 = plt.subplot(513)
    ax3.xaxis.set_visible(False)
    ax3.axhline(diff_rel_avg_s, xmax=0.01, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax3.axhline(diff_rel_max_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax3.axhline(diff_rel_min_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth, marker=None)
    plt.plot(diff_rel, label='diff_rel (=|a-b|/(|a|+|b|)): '+diff_rel_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax4 = plt.subplot(514)
    ax4.xaxis.set_visible(False)
    ax4.axhline(32, xmax=0.01, color='gray', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f32_max_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f32_min_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f32_mark_for_f16_s, xmax=1, color='red', linewidth=fig_thresh_linewidth, marker=None)
    ax4.axhline(diff_ideal_f32_man_mark_s, xmax=1, color='green', linewidth=fig_thresh_linewidth, marker=None)
    ax4.axhline(diff_ideal_f32_avg_s, xmax=0.01, color='blue', linewidth=fig_avg_linewidth, marker=None)
    plt.plot(diff_ideal_f32, label='diff_idl_f32: '+diff_ideal_f32_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    ax4 = plt.subplot(515)
    ax4.xaxis.set_visible(False)
    ax4.axhline(16, xmax=0.01, color='gray', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f16_max_s, xmax=0.02, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f16_min_s, xmax=0.02, color='green', linewidth=fig_avg_linewidth, marker=None)
    ax4.axhline(diff_ideal_f16_mark_for_f16_s, xmax=1, color='red', linewidth=fig_thresh_linewidth, marker=None)
    ax4.axhline(diff_ideal_f16_man_mark_s, xmax=1, color='green', linewidth=fig_thresh_linewidth, marker=None)
    ax4.axhline(diff_ideal_f16_avg_s, xmax=0.01, color='blue', linewidth=fig_avg_linewidth, marker=None)
    plt.plot(diff_ideal_f16, label='diff_idl_f16: '+diff_ideal_f16_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    plt.legend(loc='upper left', fontsize=fig_legend_fontsize)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f_out_fig, bbox_inches='tight')

