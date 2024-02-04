#!/usr/bin/env python3

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys
import math
import numpy as np
import torch

VER="4b"

is_debug = False

VAL_RELATIVE_SMALL_LIST_SORTED = sorted(set([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]))
assert(VAL_RELATIVE_SMALL_LIST_SORTED != 0)
VAL_RELATIVE_SMALL_LIMIT_NUM_MAX = 4096
VAL_RELATIVE_SMALL_LIMIT_RATIO_MAX = 0.25

if len(sys.argv) > 4:
    fn_g = sys.argv[1]
    fn_a = sys.argv[2]
    fn_b = sys.argv[3]
    fn_dtype = sys.argv[4]
    if len(sys.argv) > 5:
        dtype_adjust_to = sys.argv[5]
    else:
        dtype_adjust_to = None
else:
    print("Usage: G A B <pt|npy>")
    exit()

'''
print("Error: You should do following modification before run this script")
print("  1. modify the tolerance values (search 'measure: estimate tolerance')")
print("  2. remove these warning lines"); exit()
'''

if fn_dtype == "pt":
    g = torch.load(fn_g, map_location=torch.device('cpu')).detach()
    a = torch.load(fn_a, map_location=torch.device('cpu')).detach()
    b = torch.load(fn_b, map_location=torch.device('cpu')).detach()
elif fn_dtype == "npy":
    g = torch.from_numpy(np.load(fn_g))
    a = torch.from_numpy(np.load(fn_a))
    b = torch.from_numpy(np.load(fn_b))
elif fn_dtype == "bin-test":
    import tensorflow as tf
    g = torch.from_numpy(np.fromfile(fn_g, np.float32))
    a = torch.load(fn_a, map_location=torch.device('cpu')).detach()
    b = torch.from_numpy(np.fromfile(fn_b, tf.bfloat16.as_numpy_dtype))
elif fn_dtype == "bin":
    print("Not implemented file type:", fn_dtype)
    exit()
else:
    print("Not supported file type:", fn_dtype)
    exit()

# all: cast to float64
g0 = g.type(torch.float64)
a0 = a.type(torch.float64)
b0 = b.type(torch.float64)

# debug data:
if is_debug:
    #g0 = torch.tensor([0.01, 0.0, -3, 3, 0.000001, -0.25, float('inf'), +0.000020, 0, 0.0001], dtype=torch.float64)
    g0 = torch.tensor([0.01, 0.0, -3, 3, 0.000001, float('nan'), float('inf'), +0.000020, 0, float('-inf')], dtype=torch.float64)
    #g0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000020, 0, 0.0001], dtype=torch.float64)
    #a0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000022, 0, float("inf")], dtype=torch.float64)
    #a0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000021, 0, 0.0001], dtype=torch.float64)
    a0 = torch.tensor([0.01, 0.0, 0.0, 1.2, 0.000000, -0.21, 0.00005, -0.000015, 0.000002, 0.0000], dtype=torch.float64)
    b0 = torch.tensor([0.01, 0.0, -2, 0.0, 0.000000, -0.23, 0.00005, -0.000010, 0.000005, 0.0001], dtype=torch.float64)
    #b0 = torch.tensor([0.01, 0.0, -2, 1.1, float("nan"), -0.23, 0.00005, float("nan"), float("nan"), 0.0001], dtype=torch.float64)
    print("G," + ','.join('{:0.10f}'.format(i.item()) for i in g0))
    print("A," + ','.join('{:0.10f}'.format(i.item()) for i in a0))
    print("B," + ','.join('{:0.10f}'.format(i.item()) for i in b0))

def calc_mask_num(mask):
    assert(mask.max() <= 1.0)
    assert(mask.min() >= 0.0)
    return torch.sum(mask).to(torch.int).item()

# base: mark values
def calc_data_info(D):
    _zero__mask = torch.where(D == 0.0, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _zero__num = calc_mask_num(_zero__mask)
    _inf__mask = torch.where(torch.isinf(D), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _inf__num = calc_mask_num(_inf__mask)
    _nan__mask = torch.where(torch.isnan(D), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _nan__num = calc_mask_num(_nan__mask)
    _finite = torch.where(torch.isfinite(D), D, 0.0)
    _finite__max = torch.max(_finite).item()
    _finite__min = torch.min(_finite).item()
    _finite_abs = torch.abs(_finite)
    _finite_abs__avg = torch.mean(_finite_abs).item()

    if is_debug:
        print("")
        #print("D," + ','.join('{:0.10f}'.format(i.item()) for i in D))
        print("total,", D.numel())
        print("inf," + ','.join('{:0.0f}'.format(i.item()) for i in _inf__mask))
        print("nan," + ','.join('{:0.0f}'.format(i.item()) for i in _nan__mask))
        print(f"absavg,{_finite_abs__avg}")
        print(f"max,{_finite__max}")
        print(f"min,{_finite__min}")

    return _finite, _finite_abs, _finite__max, _finite__min, _finite_abs__avg, _inf__mask, _inf__num, _nan__mask, _nan__num, _zero__mask, _zero__num

def calc_data_value_info(D_abs, D_zero__mask, val_small=0.0):
    _large__mask = torch.where(D_abs >= val_small, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _large__num = calc_mask_num(_large__mask)
    _small__mask = torch.where(D_abs < val_small, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _small__mask = torch.sub(_small__mask, D_zero__mask)
    _small__num = calc_mask_num(_small__mask)
    return D_abs.numel(), _large__num, _small__num

# mark: inf nan
g1, g1_abs, g1__max, g1__min, g1_abs__avg, g1_inf__mask, g1_inf__num, g1_nan__mask, g1_nan__num, g1_zero__mask, g1_zero__num = calc_data_info(g0)
a1, a1_abs, a1__max, a1__min, a1_abs__avg, a1_inf__mask, a1_inf__num, a1_nan__mask, a1_nan__num, a1_zero__mask, a1_zero__num = calc_data_info(a0)
b1, b1_abs, b1__max, b1__min, b1_abs__avg, b1_inf__mask, b1_inf__num, b1_nan__mask, b1_nan__num, b1_zero__mask, b1_zero__num = calc_data_info(b0)

# mark: all finite
all_sum = torch.sum(torch.stack([g0, a0, b0]), dim=0)
all_finite__mask = torch.where(torch.isfinite(all_sum), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
all_finite__num = calc_mask_num(all_finite__mask)
if is_debug:
    print("all fnt,", all_finite__mask)

# mark: any zero
any_zero__mask = torch.sum(torch.stack([g1_zero__mask, a1_zero__mask, b1_zero__mask]), dim=0)
any_zero__mask = torch.mul(any_zero__mask, all_finite__mask)
any_zero__mask = torch.where(any_zero__mask > 0.0, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
any_zero__num = calc_mask_num(any_zero__mask)
if is_debug:
    print("any zero,", any_zero__mask)

# mark: all small calibrate
val_small_num_limit = min(VAL_RELATIVE_SMALL_LIMIT_NUM_MAX, all_finite__num * VAL_RELATIVE_SMALL_LIMIT_RATIO_MAX)
any_zero__mask_ = torch.mul(any_zero__mask, 3.0)
for val_small__found in VAL_RELATIVE_SMALL_LIST_SORTED:
    g1_small__mask = torch.where(g1_abs < val_small__found, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    a1_small__mask = torch.where(a1_abs < val_small__found, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    b1_small__mask = torch.where(b1_abs < val_small__found, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    any_small__mask = torch.sum(torch.stack([g1_small__mask, a1_small__mask, b1_small__mask]), dim=0)
    any_small__mask = torch.sub(any_small__mask, any_zero__mask_)
    any_small__mask = torch.mul(any_small__mask, all_finite__mask)
    any_small__mask = torch.where(any_small__mask > 0.0, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    any_small__num = calc_mask_num(any_small__mask)
    if any_small__num >= val_small_num_limit:
        break

# mark: all large
all_large__mask = torch.sub(all_finite__mask, torch.add(any_zero__mask, any_small__mask))
all_large__num = calc_mask_num(all_large__mask)
if is_debug:
    print(all_large__mask)

# mark: large small
g1_total__num, g1_larg__num, g1_omit__num = calc_data_value_info(g1_abs, g1_zero__mask, val_small=val_small__found)
a1_total__num, a1_larg__num, a1_omit__num = calc_data_value_info(a1_abs, a1_zero__mask, val_small=val_small__found)
b1_total__num, b1_larg__num, b1_omit__num = calc_data_value_info(b1_abs, b1_zero__mask, val_small=val_small__found)

#
# all: data information
print(f"_INFO,total,large,small_{val_small__found},zero,inf,nan,max,min,absavg,file")
print(f"Gold,{g1_total__num},{g1_larg__num},{g1_omit__num},{g1_zero__num},{g1_inf__num},{g1_nan__num},{g1__max:0.10f},{g1__min:0.10f},{g1_abs__avg:0.10f},{fn_g}")
print(f"A,{a1_total__num},{a1_larg__num},{a1_omit__num},{a1_zero__num},{a1_inf__num},{a1_nan__num},{a1__max:0.10f},{a1__min:0.10f},{a1_abs__avg:0.10f},{fn_a}")
print(f"B,{b1_total__num},{b1_larg__num},{b1_omit__num},{b1_zero__num},{b1_inf__num},{b1_nan__num},{b1__max:0.10f},{b1__min:0.10f},{b1_abs__avg:0.10f},{fn_b}")

# all: dtype shaping by cast
if dtype_adjust_to == "bf16":
    g1 = g1.type(torch.bfloat16).type(torch.float64)
    a1 = a1.type(torch.bfloat16).type(torch.float64)
    b1 = b1.type(torch.bfloat16).type(torch.float64)
elif dtype_adjust_to == "fp16":
    g1 = g1.type(torch.float16).type(torch.float64)
    a1 = a1.type(torch.float16).type(torch.float64)
    b1 = b1.type(torch.float16).type(torch.float64)

# func: calc adiff/rdiff
def calc_diff(A, B, pick_mask=None, pick_num=None, is_relative=False, topk_num=0):
    assert(pick_mask is not None)
    assert(pick_num is not None)
    if pick_num <= 0:
        _diff__avg, _diff_hi__num, _diff_hi__avg, _diff_lo__num, _diff_lo__avg, _diff_topk__num, _diff_topk__avg = 0.0, 0, 0.0, 0, 0.0, 0, 0.0
    else:
        if is_relative:
            _sub_abs = torch.abs(torch.sub(A, B))
            _abs_add = torch.add(torch.abs(A), torch.abs(B))
            _abs_add = torch.where(_abs_add == 0.0, 1.0, _abs_add)
            _diff = torch.div(_sub_abs, _abs_add)
        else:
            _diff = torch.sub(A, B)
            _diff = torch.abs(_diff)
        _diff = torch.where(pick_mask != 0.0, _diff, 0.0)
        _diff__avg = torch.div(torch.sum(_diff), pick_num).item()

        _diff_hi__mask = torch.where(_diff > _diff__avg, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
        _diff_hi__mask = torch.mul(_diff_hi__mask, pick_mask)
        _diff_hi__num = calc_mask_num(_diff_hi__mask)
        if _diff_hi__num <= 0:
            _diff_hi__num, _diff_hi__avg = 0, 0.0
        else:
            _diff_hi = torch.where(_diff_hi__mask != 0.0, _diff, 0.0)
            _diff_hi__avg = torch.div(torch.sum(_diff_hi), pick_num).item()

        _diff_lo__mask = torch.where(_diff <= _diff__avg, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
        _diff_lo__mask = torch.mul(_diff_lo__mask, pick_mask)
        _diff_lo__num = calc_mask_num(_diff_lo__mask)
        if _diff_lo__num <= 0:
            _diff_lo__num, _diff_lo__avg = 0, 0.0
        else:
            _diff_lo = torch.where(_diff_lo__mask != 0.0, _diff, 0.0)
            _diff_lo__avg = torch.div(torch.sum(_diff_lo), pick_num).item()

        _topk_num = topk_num
        _topk_num = _topk_num if _topk_num < _diff_hi__num else 0
        if _topk_num <= 0:
            _diff_topk__num, _diff_topk__avg = 0, 0.0
        else:
            _diff_topk, _diff_topk_arg = _diff_hi.flatten().topk(_topk_num)
            _diff_topk__avg = _diff_topk.mean()
            _diff_topk__num = _diff_topk.numel()

    return _diff__avg, _diff_hi__num, _diff_hi__avg, _diff_lo__num, _diff_lo__avg, _diff_topk__num, _diff_topk__avg

# func: compare: pass if a is less than b, or if a is not too great than b (tol=0.0 for disable tolerance)
def is_le_(a_pos, b_pos, tol_abs=0.0, tol_rel=0.0):
    assert(a_pos * b_pos >= 0)
    assert(tol_abs >= 0.0)
    assert(tol_rel >= 0.0)
    _ab_mean = (a_pos + b_pos) / 2
    if _ab_mean == 0:
        _is_passed = True
    else:
        _ab_sub = a_pos - b_pos
        _rd = _ab_sub/_ab_mean
        _is_passed_abs = (_ab_sub <= tol_abs)
        _is_passed_rel = (_rd <= tol_rel)
        _is_passed = _is_passed_abs or _is_passed_rel

    return _is_passed

# func: measure diff
def measure_diff3(G, A, B, mask, num, is_relative=False, topk_num=8, info=None, tol_abs=[0,0,0,1], tol_rel=[0,0,0,1]):
    ga__avg, ga_hi__num, ga_hi__avg, ga_lo__num, ga_lo__avg, ga_topk__num, ga_topk__avg = calc_diff(g1, a1, mask, num, is_relative=is_relative, topk_num=topk_num)
    gb__avg, gb_hi__num, gb_hi__avg, gb_lo__num, gb_lo__avg, gb_topk__num, gb_topk__avg = calc_diff(g1, b1, mask, num, is_relative=is_relative, topk_num=topk_num)
    if info is None:
        _info = "_RD" if is_relative == True else "_AD"
    else:
        _info = info
    print(f"{_info}_v{VER},avg_{num},avghi_{ga_hi__num}_{gb_hi__num},avglo_{ga_lo__num}_{gb_lo__num},top_{gb_topk__num}_{gb_topk__num}")
    print("Gold-vs-A," + ','.join('{:0.10f}'.format(i) for i in [ga__avg, ga_hi__avg, ga_lo__avg, ga_topk__avg]))
    print("Gold-vs-B," + ','.join('{:0.10f}'.format(i) for i in [gb__avg, gb_hi__avg, gb_lo__avg, gb_topk__avg]))
    is_pass_avg = is_le_(gb__avg,ga__avg, 0.0, 0.0)
    is_pass_avghi = is_le_(gb_hi__avg,ga_hi__avg, 0.0, 0.0)
    is_pass_avglo = is_le_(gb_lo__avg,ga_lo__avg, 0.0, 0.0)
    is_pass_topk = is_le_(gb_topk__avg,ga_topk__avg, 0.0, 0.0)
    if num > 3:
        is_pass_tol_avg = is_le_(gb__avg,ga__avg, tol_abs[0], tol_rel[0])
        is_pass_tol_avghi = is_le_(gb_hi__avg,ga_hi__avg, tol_abs[1], tol_rel[1])
        is_pass_tol_avglo = is_le_(gb_lo__avg,ga_lo__avg, tol_abs[2], tol_rel[2])
        is_pass_tol_topk = is_le_(gb_topk__avg,ga_topk__avg, tol_abs[3], tol_rel[3])
    else:
        is_pass_tol_avg = True
        is_pass_tol_avghi = True
        is_pass_tol_avglo = True
        is_pass_tol_topk = True

    return is_pass_avg, is_pass_avghi, is_pass_avglo, is_pass_topk, is_pass_tol_avg, is_pass_tol_avghi, is_pass_tol_avglo, is_pass_tol_topk

#
# measure: estimate tolerance

if False:  # fp16 20230927-v3
    tol_abs_zero = [0.06, 0.08, 0.06, 1]
    tol_abs_zero = [i*val_small__found for i in tol_abs_zero]
    tol_rel_zero = [0.007, 0.007, 0.015, 1]
    tol_abs_small = [0.07, 0.04, 0.04, 1]                                        
    tol_abs_small = [i*val_small__found for i in tol_abs_small]
    tol_rel_small = [0.007, 0.007, 0.01, 1]
    tol_abs_large = [0.0, 0.0, 0.0, 0.0]
    tol_rel_large = [0.008, 0.008, 0.007, 1]

if False:  # bf16 20230927-v2
    tol_abs_zero = [0.08, 0.08, 0.04, 1]
    tol_abs_zero = [i*val_small__found for i in tol_abs_zero]
    tol_rel_zero = [0.02, 0.02, 0.02, 1]
    tol_abs_small = [0.02, 0.02, 0.02, 1]
    tol_abs_small = [i*val_small__found for i in tol_abs_small]
    tol_rel_small = [0.027, 0.027, 0.05, 1]
    tol_abs_large = [0.0, 0.0, 0.0, 0.0]
    tol_rel_large = [0.008, 0.008, 0.007, 1]

if False:  # bf16 20231102-v3
    tol_abs_zero = [0.08, 0.08, 0.04, 1]
    tol_abs_zero = [i*val_small__found for i in tol_abs_zero]
    tol_rel_zero = [0.02, 0.02, 0.02, 1]
    tol_abs_small = [0.02, 0.02, 0.02, 1]
    tol_abs_small = [i*val_small__found for i in tol_abs_small]
    tol_rel_small = [0.03, 0.035, 0.05, 1]
    tol_abs_large = [0.0, 0.0, 0.0, 0.0]
    tol_rel_large = [0.01, 0.01, 0.014, 1]    

if False:    # functional 20231102-v1
    tol_abs_zero = [0.6, 0.6, 0.6, 1]
    tol_abs_zero = [i*val_small__found for i in tol_abs_zero]
    tol_rel_zero = [0.9, 0.9, 0.9, 1]
    tol_abs_small = [0.5, 0.5, 0.5, 1]
    tol_abs_small = [i*val_small__found for i in tol_abs_small]
    tol_rel_small = [0.7, 0.7, 0.7, 1]
    tol_abs_large = [0.0, 0.0, 0.0, 0.0]
    tol_rel_large = [0.3, 0.3, 0.3, 1]

if True:    # functional 20231102-v1
    tol_abs_zero = [0.0, 0.0, 0.0, 1]
    tol_abs_zero = [i*val_small__found for i in tol_abs_zero]
    tol_rel_zero = [0.0, 0.0, 0.0, 1]
    tol_abs_small = [0.0, 0.0, 0.0, 1]
    tol_abs_small = [i*val_small__found for i in tol_abs_small]
    tol_rel_small = [0.0, 0.0, 0.0, 1]
    tol_abs_large = [0.0, 0.0, 0.0, 1]
    tol_rel_large = [0.0, 0.0, 0.0, 1]

# measure: Absolute Diff for data containing zero
zero_avg, zero_avghi, zero_avglo, zero_topk, zero_avg_, zero_avghi_, zero_avglo_, zero_topk_ = measure_diff3(g1, a1, b1, any_zero__mask, any_zero__num, is_relative=False, topk_num=0, info="_AD__ZERO", tol_abs=tol_abs_zero, tol_rel=tol_rel_zero)
# measure: Absolute Diff for data containing small value
small_avg, small_avghi, small_avglo, small_topk, small_avg_, small_avghi_, small_avglo_, small_topk_ = measure_diff3(g1, a1, b1, any_small__mask, any_small__num, is_relative=False, topk_num=0, info=f"_AD_SMALL_{val_small__found}", tol_abs=tol_abs_small, tol_rel=tol_rel_small)
# measure: Relative Diff for data containing large value
large_avg, large_avghi, large_avglo, large_topk, large_avg_, large_avghi_, large_avglo_, large_topk_ = measure_diff3(g1, a1, b1, all_large__mask, all_large__num, is_relative=True, topk_num=0, info="_RD_LARGE", tol_abs=tol_abs_large, tol_rel=tol_rel_large)

# decision
is_passed_zero = zero_avg_ and zero_avghi_ and zero_avglo_ and zero_topk_
is_passed_small = small_avg_ and small_avghi_ and small_avglo_ and small_topk_
is_passed_large = large_avg_ and large_avghi_ and large_avglo_ and large_topk_
is_need_warn_zero = not zero_avg or not zero_avghi or not zero_avglo
is_need_warn_small = not small_avg or not small_avghi or not small_avglo
is_need_warn_large = not large_avg or not large_avghi or not large_avglo

if is_passed_zero and is_passed_small and is_passed_large:
    if is_need_warn_zero or is_need_warn_small or is_need_warn_large:
        result_info = "PASSWARN"
    else:
        result_info = "PASS"
else:
    result_info = "FAIL"

print(f"result_v{VER}={result_info},zero={zero_avg:d}{zero_avghi:d}{zero_avglo:d}{zero_topk:d}_{zero_avg_:d}{zero_avghi_:d}{zero_avglo_:d}{zero_topk_:d}_{any_zero__num},small={small_avg:d}{small_avghi:d}{small_avglo:d}{small_topk:d}_{small_avg_:d}{small_avghi_:d}{small_avglo_:d}{small_topk_:d}_{any_small__num},large={large_avg:d}{large_avghi:d}{large_avglo:d}{large_topk:d}_{large_avg_:d}{large_avghi_:d}{large_avglo_:d}{large_topk_:d}_{all_large__num}")
