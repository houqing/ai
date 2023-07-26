#!/usr/bin/env python3

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys

import numpy as np

VER=16

def usage_exit(err_info='', err_no=-1):
    if err_info:
        print('Error: ', err_info)
    print('Usage:', sys.argv[0], '<file_a> <file_b> <fp16|fp32|npy>-<fp16|fp32|npy>')
    exit(err_no)

if len(sys.argv) < 4:
    usage_exit()

output_info = []
def build_out_info(info_list, info):
    info_list.append(info)
    print(info)

# param define
is_need_cast_input_to_fp16 = False
is_ideal_need_align_dtype = False
is_calc_cd = False
is_calc_jsd = False
is_calc_inc = True
is_calc_rel = True
is_calc_b16 = True
is_calc_b32 = False
is_random = False
is_out_no_log = False
is_out_no_fig = False
is_marker_factor_auto = False
TODO_is_bit_factor_auto = False
is_sort = False
conf_seg_off = None
conf_seg_len = None
conf_fig_format = 'png'
if len(sys.argv) >= 5:
    for arg in sys.argv[4:]:
        if arg in [ 'sort' ]:
            is_sort = True
        elif arg in [ 'nofig', 'nopic' ]:
            is_out_no_fig = True
        elif arg in [ 'nolog' ]:
            is_out_no_log = True
        elif arg in [ 'nofile' ]:
            is_out_no_fig = True
            is_out_no_log = True
        elif arg in [ 'cd', 'cosd', 'cosined' ]:
            is_calc_cd = True
        elif arg in [ 'jsd', 'jensen_shannond' ]:
            is_calc_jsd = True
        elif arg in [ 'noinc' ]:
            is_calc_inc = False
        elif arg in [ 'norel' ]:
            is_calc_rel = False
        elif arg in [ 'nob16' ]:
            is_calc_b16 = False
        elif arg in [ 'nob32' ]:
            is_calc_b32 = False
        elif arg in [ 'b32' ]:
            is_calc_b32 = True
        # XXX debug only options, bugs are features here
        elif arg in [ 'ran', 'rand', 'random' ]:
            is_random = True
        elif arg in [ 'png', 'jpg', 'jpeg', 'svg', 'pdf' ]:
            conf_fig_format = arg
        elif arg in [ 'autodot', 'automarker' ]:
            is_marker_factor_auto = True
        elif arg in [ 'autobit', 'autoidl', 'autoideal' ]:
            TODO_is_bit_factor_auto = True
        elif arg in [ 'f16', 'fp16' ]:
            is_need_cast_input_to_fp16 = True
        elif arg in [ 'ideal-align-dtype', 'align-dtype', 'align' ]:
            is_ideal_need_align_dtype = True
        elif arg.startswith('offset=') or arg.startswith('off='):
            conf_seg_off = int(arg.lstrip('offset='))
        elif arg.startswith('length=') or arg.startswith('len='):
            conf_seg_len = int(arg.lstrip('length='))
        else:
            usage_exit('unknown parameter "'+arg+'"')

# init
if not is_random:
    np.random.seed(0x1234)
    
# get input
f_a = sys.argv[1]
f_b = sys.argv[2]
my_is_same_ab = False
if f_a == f_b:
    my_is_same_ab = True
f_out_log = f_b + '--sort-diff.log' if is_sort else f_b + '--diff.log'
f_out_fig = f_b + '--sort-diff.' + conf_fig_format if is_sort else f_b + '--diff.' + conf_fig_format
_f_type = sys.argv[3]
f_type = _f_type.split('-')

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
    a = np.load(f_a)
    a_t = 'n'+str(a.dtype)
    a_dtype = a.dtype
else:
    usage_exit()

if not my_is_same_ab:
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
        b = np.load(f_b)
        b_t = 'n'+str(b.dtype)
        b_dtype = b.dtype
    else:
        usage_exit()
else:
    if np.issubdtype(a.dtype, np.floating):
        b = a.astype(np.float16)
    else:
        b = a
    b_t = a_t
    b_dtype = a_dtype

# generate output info head
f_a_info = 'A_'+a_t+'--v'+str(VER)+': '+f_a
f_b_info = 'B_'+b_t+'--v'+str(VER)+': '+f_b
build_out_info(output_info, f_a_info)
build_out_info(output_info, f_b_info)
if not is_out_no_log:
    build_out_info(output_info, 'log   : ' + f_out_log)
if not is_out_no_fig:
    build_out_info(output_info, 'fig   : ' + f_out_fig)

def get_shape_str(shape):
    shape_str = '_'.join(str(i) for i in shape)
    if shape_str == '':
        shape_str = 's'
    return shape_str

st_a_total_raw = a.size
st_b_total_raw = b.size
st_a_shape_raw = get_shape_str(a.shape)
st_b_shape_raw = get_shape_str(b.shape)
if st_a_shape_raw != st_b_shape_raw or st_a_total_raw != st_b_total_raw:
    st_a_shape_raw = '__' + st_a_shape_raw + '__'
    st_b_shape_raw = '__' + st_b_shape_raw + '__'

a = a.reshape(-1)
b = b.reshape(-1)

ab_min_dtype = np.float32
ab_max_dtype = np.float16
if a_dtype == np.float16 or b_dtype == np.float16:
    ab_min_dtype = np.float16
if a_dtype == np.float32 or b_dtype == np.float32:
    ab_max_dtype = np.float32

if is_ideal_need_align_dtype:
    if ab_max_dtype == np.float16 or is_need_cast_input_to_fp16:
        is_need_cast_input_to_fp16 = True

# fix input
st_a_total_aligned = st_a_total_raw
st_b_total_aligned = st_b_total_raw
if st_a_total_raw > st_b_total_raw:
    b = np.append(b, np.zeros(st_a_total_raw - st_b_total_raw, b_dtype))
    st_b_total_aligned = st_a_total_raw
else:
    a = np.append(a, np.zeros(st_b_total_raw - st_a_total_raw, a_dtype))
    st_a_total_aligned = st_b_total_raw

# data random pick
st_pick_data_total = st_a_total_aligned
my_is_data_pick = False
if len(f_type) == 3:
    _arg_pick = np.random.permutation(st_pick_data_total)
    st_pick_data_total = max(min(st_pick_data_total, int(f_type[2])), 1)
    _arg_pick = _arg_pick[:st_pick_data_total]
    _arg_pick.sort()
    a = np.take(a, _arg_pick)
    b = np.take(b, _arg_pick)
    my_is_data_pick = True
st_a_pick_data_total = len(a)
st_b_pick_data_total = len(b)

# data sort
if is_sort:
    _arg_sort = np.argsort(a)
    a = np.take(a, _arg_sort)
    b = np.take(b, _arg_sort)

# data seg
_seg_begin = 0
_seg_end = st_pick_data_total
my_is_data_seg = False
if conf_seg_off is not None and conf_seg_off > 0:
    if conf_seg_off < st_pick_data_total:
        _seg_begin = conf_seg_off
    my_is_data_seg = True
if conf_seg_len is not None and conf_seg_len > 0:
    _seg_end = _seg_begin + conf_seg_len
    if _seg_end > st_pick_data_total:
        _seg_end = st_pick_data_total
    my_is_data_seg = True
a = a[_seg_begin:_seg_end]
b = b[_seg_begin:_seg_end]
st_a_seg_data_total = len(a)
st_b_seg_data_total = len(b)
st_seg_data_total = st_a_seg_data_total


# precision fix
if is_need_cast_input_to_fp16:
    a = a.astype(np.float16)
    b = b.astype(np.float16)
a = a.astype(np.float64)
b = b.astype(np.float64)


# calc stat
_st_a_inf = len(np.argwhere(np.isinf(a)))
_st_a_nan = len(np.argwhere(np.isnan(a)))
_st_a_zero = len(np.argwhere(np.equal(a, 0)))
_st_a_pos = len(np.argwhere(np.greater(a, 0)))
_st_a_neg = len(np.argwhere(np.less(a, 0)))
_st_b_inf = len(np.argwhere(np.isinf(b)))
_st_b_nan = len(np.argwhere(np.isnan(b)))
_st_b_zero = len(np.argwhere(np.equal(b, 0)))
_st_b_pos = len(np.argwhere(np.greater(b, 0)))
_st_b_neg = len(np.argwhere(np.less(b, 0)))

_st_zamb = _st_a_zero - _st_b_zero
_st_pamb = _st_a_pos - _st_b_pos
_st_namb = _st_a_neg - _st_b_neg
_st_zamb_info = ' zamb='+str(_st_zamb)
_st_zbma_info = _st_zamb_info
_st_pamb_info = ' pamb='+str(_st_pamb)
_st_pbma_info = _st_pamb_info
_st_namb_info = ' namb='+str(_st_namb)
_st_nbma_info = _st_namb_info
_st_zapb = _st_a_zero + _st_b_zero
_st_zabd = (abs(_st_zamb) / _st_zapb) if _st_zapb != 0 else _st_zapb
_st_zabd = float(_st_zabd)
_st_zambd_info = ' zabd='+str(_st_zabd)
_st_zbmad_info = _st_zambd_info
_st_zdp = abs(_st_zamb) / st_seg_data_total
_st_zdp_info = ' zdp='+str(_st_zdp)
_st_pdp = abs(_st_pamb) / st_seg_data_total
_st_pdp_info = ' pdp='+str(_st_pdp)
_st_ndp = abs(_st_namb) / st_seg_data_total
_st_ndp_info = ' ndp='+str(_st_ndp)

_st_a_pick_info = str(st_a_pick_data_total)+'#' if my_is_data_pick else ''
_st_a_seg_info = str(st_a_seg_data_total)+'@'+str(_seg_begin)+'/'+str(_seg_end)+'#' if my_is_data_seg else ''
st_a_info = 't='+_st_a_seg_info+_st_a_pick_info+str(st_a_total_raw)+' s='+str(st_a_shape_raw)+' inf='+str(_st_a_inf)+' nan='+str(_st_a_nan)+' z='+str(_st_a_zero)+' p='+str(_st_a_pos)+' n='+str(_st_a_neg)+_st_zamb_info+_st_pamb_info+_st_namb_info+_st_zambd_info+_st_zdp_info+_st_pdp_info+_st_ndp_info
build_out_info(output_info, 'info_a: ' + st_a_info)
_st_b_pick_info = str(st_b_pick_data_total)+'#' if my_is_data_pick else ''
_st_b_seg_info = str(st_b_seg_data_total)+'@'+str(_seg_begin)+'/'+str(_seg_end)+'#' if my_is_data_seg else ''
st_b_info = 't='+_st_b_seg_info+_st_b_pick_info+str(st_b_total_raw)+' s='+str(st_b_shape_raw)+' inf='+str(_st_b_inf)+' nan='+str(_st_b_nan)+' z='+str(_st_b_zero)+' p='+str(_st_b_pos)+' n='+str(_st_b_neg)+_st_zbma_info+_st_pbma_info+_st_nbma_info+_st_zambd_info+_st_zdp_info+_st_pdp_info+_st_ndp_info
build_out_info(output_info, 'info_b: ' + st_b_info)


# process input
g_aa = a
g_bb = b


def calc_canb_dist_elemwise(data_a, data_b):
    _abs_sum = np.abs(data_a) + np.abs(data_b)
    _sub_abs = np.abs(data_a - data_b)
    _arg_zeros = np.argwhere(np.equal(_abs_sum, 0))
    np.put(_abs_sum, _arg_zeros, 1)
    _diff_rel = _sub_abs / _abs_sum
    np.put(_diff_rel, _arg_zeros, 0)
    return _diff_rel

def calc_cosine_dist(data_a, data_b):
    if False:
        import scipy.spatial as sp_spatial
        _cd = sp_spatial.distance.cosine(data_a, data_b)
    else:
        _eps = np.array(1, np.uint64).view(np.float64)
        uv = np.mean(data_a * data_b)
        uu = np.mean(np.square(data_a))
        vv = np.mean(np.square(data_b))
        _cd = 1.0 - (uv + _eps) / (np.sqrt(uu * vv) + _eps)
    return _cd

def calc_js_dist(data_a, data_b):
    import scipy.spatial as sp_spatial
    import scipy.special as sp_special
    _a_sm = sp_special.softmax(data_a)
    _b_sm = sp_special.softmax(data_b)
    _js = sp_spatial.distance.jesenshannon(_a_sm, _b_sm)
    return _js

# function to generate data averages
def gen_max_min_avg(data, mode='all'):  # mode:all|avg|avg2|pos|neg|pos2|neg2|pos4|neg4
    if mode in ['pos', 'pos2', 'pos4']:
        _arg_data = np.argwhere(np.greater(data, 0))
        _d = np.take(data, _arg_data).reshape(-1)
    elif mode in ['neg', 'neg2', 'neg4']:
        _arg_data = np.argwhere(np.less(data, 0))
        _d = np.take(data, _arg_data).reshape(-1)
    elif mode in ['all', 'avg', 'avg2']:
        _d = data
    else:
        usage_exit('Error: bad mode ' + mode)

    if len(_d) == 0:
        _d = [0]
    if mode == 'pos2':
        _min = np.min(_d)
        _avg = np.mean(_d)
        return _min, _avg
    elif mode == 'neg2':
        _max = np.max(_d)
        _avg = np.mean(_d)
        return _max, _avg
    elif mode == 'pos4':
        _max = np.max(_d)
        _min = np.min(_d)
        _avg = np.mean(_d)
        _arg_data = np.argwhere(np.greater_equal(_d, _avg))
        _dmaxa = np.take(_d, _arg_data).reshape(-1)
        if len(_dmaxa) == 0:
            _dmaxa = [0]
        _avg_max = np.mean(_dmaxa)
        _arg_data = np.argwhere(np.less(_d, _avg))
        _dmina = np.take(_d, _arg_data).reshape(-1)
        if len(_dmina) == 0:
            _dmina = [0]
        _avg_min = np.mean(_dmina)
        return _max, _min, _avg, _avg_max, _avg_min
    elif mode == 'neg4':
        _max = np.max(_d)
        _min = np.min(_d)
        _avg = np.mean(_d)
        _arg_data = np.argwhere(np.less_equal(_d, _avg))
        _dmina = np.take(_d, _arg_data).reshape(-1)
        if len(_dmina) == 0:
            _dmina = [0]
        _avg_min = np.mean(_dmina)
        _arg_data = np.argwhere(np.greater(_d, _avg))
        _dmaxa = np.take(_d, _arg_data).reshape(-1)
        if len(_dmaxa) == 0:
            _dmaxa = [0]
        _avg_max = np.mean(_dmaxa)
        return _max, _min, _avg, _avg_max, _avg_min
    elif mode == 'avg':
        _avg = np.mean(_d)
        return _avg
    elif mode == 'avg2':
        _avg = np.mean(_d)
        _arg_data = np.argwhere(np.not_equal(data, 0))
        _dnz = np.take(data, _arg_data).reshape(-1)
        if len(_dnz) == 0:
            _dnz = [0]
        _avg_nz = np.mean(_dnz)
        return _avg, _avg_nz
    else:
        _max = np.max(_d)
        _min = np.min(_d)
        _avg = np.mean(_d)
        return _max, _min, _avg


# remove inf/nan before calc
_arg_aa_inf_nan = np.argwhere(np.logical_or(np.isinf(g_aa), np.isnan(g_aa)))
_arg_bb_inf_nan = np.argwhere(np.logical_or(np.isinf(g_bb), np.isnan(g_bb)))
np.put(g_aa, _arg_aa_inf_nan, 0)
np.put(g_bb, _arg_bb_inf_nan, 0)


# calc data statistic
g_aa_avg_s, g_aa_avg_nz_s = gen_max_min_avg(g_aa, mode='avg2')
g_aa_pos_max_s, g_aa_pos_min_s, g_aa_pos_avg_s, g_aa_pos_max_avg_s, g_aa_pos_min_avg_s = gen_max_min_avg(g_aa, mode='pos4')
g_aa_neg_max_s, g_aa_neg_min_s, g_aa_neg_avg_s, g_aa_neg_max_avg_s, g_aa_neg_min_avg_s = gen_max_min_avg(g_aa, mode='neg4')
g_data_a_info = 'pmax='+str(g_aa_pos_max_s)+' nmin='+str(g_aa_neg_min_s)+' avg='+str(g_aa_avg_s)+' avgnz='+str(g_aa_avg_nz_s)+' pavg='+str(g_aa_pos_avg_s)+' navg='+str(g_aa_neg_avg_s)+' pmin='+str(g_aa_pos_min_s)+' nmax='+str(g_aa_neg_max_s)
g_data_a_info = g_data_a_info+' pmaxa='+str(g_aa_pos_max_avg_s)+' pmina='+str(g_aa_pos_min_avg_s)+' nmaxa='+str(g_aa_neg_max_avg_s)+' nmina='+str(g_aa_neg_min_avg_s)
build_out_info(output_info, 'data_a: ' + g_data_a_info)

g_bb_avg_s, g_bb_avg_nz_s = gen_max_min_avg(g_bb, mode='avg2')
g_bb_pos_max_s, g_bb_pos_min_s, g_bb_pos_avg_s, g_bb_pos_max_avg_s, g_bb_pos_min_avg_s = gen_max_min_avg(g_bb, mode='pos4')
g_bb_neg_max_s, g_bb_neg_min_s, g_bb_neg_avg_s, g_bb_neg_max_avg_s, g_bb_neg_min_avg_s = gen_max_min_avg(g_bb, mode='neg4')
g_data_b_info = 'pmax='+str(g_bb_pos_max_s)+' nmin='+str(g_bb_neg_min_s)+' avg='+str(g_bb_avg_s)+' avgnz='+str(g_bb_avg_nz_s)+' pavg='+str(g_bb_pos_avg_s)+' navg='+str(g_bb_neg_avg_s)+' pmin='+str(g_bb_pos_min_s)+' nmax='+str(g_bb_neg_max_s)
g_data_b_info = g_data_b_info+' pmaxa='+str(g_bb_pos_max_avg_s)+' pmina='+str(g_bb_pos_min_avg_s)+' nmaxa='+str(g_bb_neg_max_avg_s)+' nmina='+str(g_bb_neg_min_avg_s)
build_out_info(output_info, 'data_b: ' + g_data_b_info)


# calc incremental diff
if is_calc_inc:
    g_diff_inc = g_aa - g_bb
    g_diff_inc_pos_max_s, g_diff_inc_pos_min_s, g_diff_inc_pos_avg_s = gen_max_min_avg(g_diff_inc, mode='pos')
    g_diff_inc_neg_max_s, g_diff_inc_neg_min_s, g_diff_inc_neg_avg_s = gen_max_min_avg(g_diff_inc, mode='neg')
    _arg_non_zeros = np.argwhere(np.not_equal(g_diff_inc, 0.0))
    diff_inc_diff_num = len(_arg_non_zeros)
    g_diff_info = 'diff_num='+str(diff_inc_diff_num)+' pmax='+str(g_diff_inc_pos_max_s)+' nmin='+str(g_diff_inc_neg_min_s)+' pavg='+str(g_diff_inc_pos_avg_s)+' navg='+str(g_diff_inc_neg_avg_s)+' pmin='+str(g_diff_inc_pos_min_s)+' nmax='+str(g_diff_inc_neg_max_s)
    build_out_info(output_info, 'diff_inc: ' + g_diff_info)

# calc canberra dist
if is_calc_rel:
    g_diff_rel = calc_canb_dist_elemwise(g_aa, g_bb)
    g_diff_rel_max_s, g_diff_rel_min_s, g_diff_rel_avg_s = gen_max_min_avg(g_diff_rel, mode='all')
    g_diff_rel_pos_min_s, g_diff_rel_pos_avg_s = gen_max_min_avg(g_diff_rel, mode='pos2')
    g_diff_rel_pos_max_s = g_diff_rel_max_s
    diff_rel_info = 'avg='+str(g_diff_rel_avg_s)+' max='+str(g_diff_rel_max_s)+' min='+str(g_diff_rel_min_s)+' pavg='+str(g_diff_rel_pos_avg_s)+' pmax='+str(g_diff_rel_pos_max_s)+' pmin='+str(g_diff_rel_pos_min_s)
    # calc consine dist
    if is_calc_cd:
        diff_cosine_dist = calc_cosine_dist(g_aa, g_bb)
        diff_rel_info = diff_rel_info + ' cd='+str(diff_cosine_dist)
    # calc js dist
    if is_calc_jsd:
        diff_js_dist = calc_js_dist(g_aa, g_bb)
        diff_rel_info = diff_rel_info + ' jsd='+str(diff_js_dist)
    build_out_info(output_info, 'diff_rel: ' + diff_rel_info)

# calc ideal diff
def gen_ideal_diff(data_a, data_b, dtype=np.float32):
    if dtype in [ np.float32 ]:
        _AB_dtype_f = np.float32
        _AB_dtype_u = np.uint32
        _AB_u_abs_mask = 0x7fffffff
        _AB_u_sign_mask = 0x80000000
        _AB_u_sign_shift = 31
    elif dtype in [ np.float16 ]:
        _AB_dtype_f = np.float16
        _AB_dtype_u = np.uint16
        _AB_u_abs_mask = 0x7fff
        _AB_u_sign_mask = 0x80000000
        _AB_u_sign_shift = 15

    A_f = data_a.astype(_AB_dtype_f)
    A_u = A_f.view(_AB_dtype_u)
    A_u64_abs = (A_u & _AB_u_abs_mask).astype(np.uint64)
    A_u_sign = (A_u & _AB_u_sign_mask) >> _AB_u_sign_shift
    B_f = data_b.astype(_AB_dtype_f)
    B_u = B_f.view(_AB_dtype_u)
    B_u64_abs = (B_u & _AB_u_abs_mask).astype(np.uint64)
    B_u_sign = (B_u & _AB_u_sign_mask) >> _AB_u_sign_shift

    _AB_u_sign_is_different = np.logical_xor(A_u_sign, B_u_sign)
    _AB_u_sign_is_same = np.logical_not(_AB_u_sign_is_different)

    # calc for different sign
    _AB_u64_diff_val_for_diff_sign = A_u64_abs + B_u64_abs + 1
    # calc for same sign
    _AB_u64_diff_val_for_same_sign = np.maximum(A_u64_abs, B_u64_abs) - np.minimum(A_u64_abs, B_u64_abs) + 1
    # pick valid diff/same sign value
    _AB_u64_diff_val = np.where(_AB_u_sign_is_same, _AB_u64_diff_val_for_same_sign, _AB_u64_diff_val_for_diff_sign)
    # calc the equiv bits
    AB_diff_bits = np.log2(_AB_u64_diff_val)

    def _add_factor(_d, _is_add=False):
        if _is_add:
            return _d   # TODO
        else:
            return _d

    _diff_ideal_thresh_s = _add_factor(_AB_u_sign_shift + 1, False)
    _diff_ideal_a_ref = _add_factor(np.log2(A_u64_abs + 1), TODO_is_bit_factor_auto)
    _diff_ideal = _add_factor(AB_diff_bits, TODO_is_bit_factor_auto)

    return _diff_ideal, _diff_ideal_thresh_s, _diff_ideal_a_ref

if is_calc_b16:
    g_diff_ideal_f16, g_diff_ideal_f16_thresh_s, g_diff_ideal_f16_a_ref = gen_ideal_diff(g_aa, g_bb, np.float16)
    g_diff_ideal_f16_max_s, g_diff_ideal_f16_min_s, g_diff_ideal_f16_avg_s = gen_max_min_avg(g_diff_ideal_f16, mode='all')
    g_diff_ideal_f16_pos_min_s, g_diff_ideal_f16_pos_avg_s = gen_max_min_avg(g_diff_ideal_f16, mode='pos2')
    g_diff_ideal_f16_pos_max_s = g_diff_ideal_f16_max_s
    diff_ideal_f16_info = 'avg='+str(g_diff_ideal_f16_avg_s)+' max='+str(g_diff_ideal_f16_max_s)+' min='+str(g_diff_ideal_f16_min_s)+' pavg='+str(g_diff_ideal_f16_pos_avg_s)+' pmax='+str(g_diff_ideal_f16_pos_max_s)+' pmin='+str(g_diff_ideal_f16_pos_min_s)
    build_out_info(output_info, 'diff_b16: ' + diff_ideal_f16_info)

if is_calc_b32:
    g_diff_ideal_f32, g_diff_ideal_f32_thresh_s, g_diff_ideal_f32_a_ref = gen_ideal_diff(g_aa, g_bb, np.float32)
    g_diff_ideal_f32_max_s, g_diff_ideal_f32_min_s, g_diff_ideal_f32_avg_s = gen_max_min_avg(g_diff_ideal_f32, mode='all')
    g_diff_ideal_f32_pos_min_s, g_diff_ideal_f32_pos_avg_s = gen_max_min_avg(g_diff_ideal_f32, mode='pos2')
    g_diff_ideal_f32_pos_max_s = g_diff_ideal_f32_max_s
    diff_ideal_f32_info = 'avg='+str(g_diff_ideal_f32_avg_s)+' max='+str(g_diff_ideal_f32_max_s)+' min='+str(g_diff_ideal_f32_min_s)+' pavg='+str(g_diff_ideal_f32_pos_avg_s)+' pmax='+str(g_diff_ideal_f32_pos_max_s)+' pmin='+str(g_diff_ideal_f32_pos_min_s)
    build_out_info(output_info, 'diff_b32: ' + diff_ideal_f32_info)


# output text
if not is_out_no_log:
    with open(f_out_log, 'w') as f:
        for i in output_info:
            f.write(i + '\n')


# output figure
if not is_out_no_fig:
    import matplotlib.pyplot as plt
    # output figure
    fig_title = f_a_info + '\n' + f_b_info
    fig_avg_linewidth = 0.5
    fig_thresh_linewidth = 0.05
    fig_alpha = 0.5
    if is_marker_factor_auto:
        _fig_markersize_factor = max(0.11, (30 / np.log(st_seg_data_total + 1) + -1.55))
    else:
        _fig_markersize_factor = 1
    fig_markersize_a = 1.6 * _fig_markersize_factor
    fig_markersize_b = 1.5 * _fig_markersize_factor
    fig_markersize_mod = 1.5 * _fig_markersize_factor
    fig_markersize = 1.5 * _fig_markersize_factor
    fig_legend_fontsize = 'xx-small'
    fig_legend_alpha = 0.6
    fig_legend_label_color = 'lightgray'

    _subg_mark = [1, is_calc_inc, is_calc_rel, is_calc_b16, is_calc_b32]
    _subg_num = sum(_subg_mark)
    _subg_id_min = _subg_num * 100 + 11
    _subg_id_max = _subg_id_min + _subg_num
    _subg_id_list = list(range(_subg_id_min, _subg_id_max))

    plt.figure(1, figsize=(23, 11))

    _subg_id_list_id = 0
    v = _subg_id_list[_subg_id_list_id]
    ax = plt.subplot(v, xbound=110)
    ax.xaxis.tick_top()
    ax.axhline(g_aa_pos_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_aa_pos_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_aa_pos_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_aa_neg_max_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_aa_neg_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_aa_neg_min_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_neg_max_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_neg_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    ax.axhline(g_bb_neg_min_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
    plt.title(fig_title, loc='left', fontdict={'fontsize': 8})
    plt.plot(g_aa, label='data_a: '+st_a_info+' '+g_data_a_info, linewidth=0, marker='.', markersize=fig_markersize_a, markeredgewidth=0, markerfacecolor='red', alpha=fig_alpha)
    plt.plot(g_bb, label='data_b: '+st_b_info+' '+g_data_b_info, linewidth=0, marker='.', markersize=fig_markersize_b, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
    leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
    leg.get_frame().set_alpha(fig_legend_alpha)
    plt.setp(leg.get_texts(), color=fig_legend_label_color)

    if is_calc_inc:
        _subg_id_list_id += 1
        v = _subg_id_list[_subg_id_list_id]
        ax = plt.subplot(v)
        ax.xaxis.set_visible(False)
        ax.axhline(g_diff_inc_pos_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_inc_pos_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_inc_pos_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_inc_neg_max_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_inc_neg_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_inc_neg_min_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
        plt.plot(g_diff_inc, label='diff_inc (=a-b): '+g_diff_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
        leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
        leg.get_frame().set_alpha(fig_legend_alpha)
        plt.setp(leg.get_texts(), color=fig_legend_label_color)

    if is_calc_rel:
        _subg_id_list_id += 1
        v = _subg_id_list[_subg_id_list_id]
        ax = plt.subplot(v)
        ax.xaxis.set_visible(False)
        ax.axhline(g_diff_rel_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_rel_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_rel_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_rel_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_rel_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_rel_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        plt.plot(g_diff_rel, label='diff_rel (=|a-b|/(|a|+|b|)): '+diff_rel_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
        leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
        leg.get_frame().set_alpha(fig_legend_alpha)
        plt.setp(leg.get_texts(), color=fig_legend_label_color)

    if is_calc_b16:
        _subg_id_list_id += 1
        v = _subg_id_list[_subg_id_list_id]
        ax = plt.subplot(v)
        ax.xaxis.set_visible(False)
        ax.axhline(g_diff_ideal_f16_thresh_s, xmin=0.000, xmax=0.010, color='gray', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f16_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        plt.plot(g_diff_ideal_f16_a_ref, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='lightgray', alpha=fig_alpha)
        plt.plot(g_diff_ideal_f16, label='diff_b16: '+diff_ideal_f16_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
        leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
        leg.get_frame().set_alpha(fig_legend_alpha)
        plt.setp(leg.get_texts(), color=fig_legend_label_color)

    if is_calc_b32:
        _subg_id_list_id += 1
        v = _subg_id_list[_subg_id_list_id]
        ax = plt.subplot(v)
        ax.xaxis.set_visible(False)
        ax.axhline(g_diff_ideal_f32_thresh_s, xmin=0.000, xmax=0.010, color='gray', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_max_s, xmin=0.000, xmax=0.020, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_avg_s, xmin=0.000, xmax=0.015, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_min_s, xmin=0.000, xmax=0.010, color='red', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_pos_max_s, xmin=0.020, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_pos_avg_s, xmin=0.025, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        ax.axhline(g_diff_ideal_f32_pos_min_s, xmin=0.030, xmax=0.040, color='blue', linewidth=fig_avg_linewidth, marker=None)
        plt.plot(g_diff_ideal_f32_a_ref, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='lightgray', alpha=fig_alpha)
        plt.plot(g_diff_ideal_f32, label='diff_b32: '+diff_ideal_f32_info, linewidth=0, marker='.', markersize=fig_markersize, markeredgewidth=0, markerfacecolor='blue', alpha=fig_alpha)
        leg = plt.legend(loc='upper left', frameon=True, fontsize=fig_legend_fontsize)
        leg.get_frame().set_alpha(fig_legend_alpha)
        plt.setp(leg.get_texts(), color=fig_legend_label_color)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f_out_fig, bbox_inches='tight', transparent=True)
