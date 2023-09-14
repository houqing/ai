#!/usr/bin/env python3                                                                                                  

#
# houqing@(Turing Architecture and Design Dept, HS)
#

import sys
import numpy as np
import torch

is_debug = False

VAL_RELATIVE_SMALL = 0.0000001
VAL_RELATIVE_SMALL = 0.0001
#VAL_RELATIVE_SMALL = 0.0


torch.set_printoptions(precision=8, sci_mode=False, linewidth=1024)
np.set_printoptions(precision=8, suppress=True, linewidth=1024)

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
    print("Usage: G A B <pt|npy> [bf16|fp16]")
    exit()

if fn_dtype == "pt":
    g = torch.load(fn_g, map_location=torch.device('cpu')).detach()
    a = torch.load(fn_a, map_location=torch.device('cpu')).detach()
    b = torch.load(fn_b, map_location=torch.device('cpu')).detach()
elif fn_dtype == "npy":
    g = torch.from_numpy(np.load(fn_g))
    a = torch.from_numpy(np.load(fn_a))
    b = torch.from_numpy(np.load(fn_b))
elif fn_dtype == "bin-test":
    g = torch.from_numpy(np.fromfile(fn_g, np.float32))
    a = torch.from_numpy(np.fromfile(fn_a, np.float16))
    b = torch.from_numpy(np.fromfile(fn_b, np.float16))
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

# fake data:
if is_debug:
    #g0 = torch.tensor([0.01, 0.0, -3, 3, 0.000001, -0.25, float('inf'), +0.000020, 0, 0.0001], dtype=torch.float64)
    g0 = torch.tensor([0.01, 0.0, -3, 3, 0.000001, float('nan'), float('inf'), +0.000020, 0, float('-inf')], dtype=torch.float64)
    #g0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000020, 0, 0.0001], dtype=torch.float64)
    #a0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000022, 0, float("inf")], dtype=torch.float64)
    #a0 = torch.tensor([0.01, 0.0, -3, 1, 0.000001, -0.25, 0.00005, +0.000021, 0, 0.0001], dtype=torch.float64)
    a0 = torch.tensor([0.01, 0.0, 0.0, 1.2, 0.000000, -0.21, 0.00005, -0.000015, 0.000002, 0.0001], dtype=torch.float64)
    b0 = torch.tensor([0.01, 0.0, -2, 0.0, 0.000000, -0.23, 0.00005, -0.000010, 0.000005, 0.0001], dtype=torch.float64)
    #b0 = torch.tensor([0.01, 0.0, -2, 1.1, float("nan"), -0.23, 0.00005, float("nan"), float("nan"), 0.0001], dtype=torch.float64)

def calc_mask_num(mask):
    assert(mask.max() <= 1.0)
    assert(mask.min() >= 0.0)
    return torch.sum(mask).to(torch.int).item()

# all: mark values
all_abs = torch.sum(torch.stack([torch.abs(g0), torch.abs(a0), torch.abs(b0)]), dim=0)
all_zero__mask = torch.where(all_abs == 0.0, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
all_finite__mask = torch.where(torch.isfinite(all_abs), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))

# base: mark values
def calc_base_info(D):
    #D = D.flatten()
    _abs = torch.abs(D)
    _zero__mask = torch.where(_abs == 0.0, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _inf__mask = torch.where(torch.isinf(D), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _nan__mask = torch.where(torch.isnan(D), torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _larg_pos__mask = torch.where(D > VAL_RELATIVE_SMALL, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _larg_neg__mask = torch.where(D < -VAL_RELATIVE_SMALL, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    _larg__mask = torch.add(_larg_pos__mask, _larg_neg__mask)
    _larg__mask = torch.sub(_larg__mask, _inf__mask)  # XXX exclude inf from large
    _mark__mask = torch.sum(torch.stack([_larg__mask, _zero__mask, _inf__mask, _nan__mask]), dim=0)
    _omit__mask = torch.sub(1.0, _mark__mask)
    assert(_omit__mask.max() <= 1.0)
    assert(_omit__mask.min() >= 0.0)
    _total__num = D.numel()
    _larg__num = calc_mask_num(_larg__mask)
    _zero__num = calc_mask_num(_zero__mask)
    _inf__num = calc_mask_num(_inf__mask)
    _nan__num = calc_mask_num(_nan__mask)
    _omit__num = calc_mask_num(_omit__mask)
    _max = torch.max(D)
    _min = torch.min(D)
    _absavg = torch.mean(_abs)

    if is_debug:
    #if True:
        print("")
        #print("D," + ','.join('{:0.8f}'.format(i.numpy()) for i in D))
        print("total," + D.numel())
        print("larg," + ','.join('{:0.0f}'.format(i.numpy()) for i in _larg__mask))
        print("omit," + ','.join('{:0.0f}'.format(i.numpy()) for i in _omit__mask), f"(>{VAL_RELATIVE_SMALL})")
        print("zero," + ','.join('{:0.0f}'.format(i.numpy()) for i in _zero__mask))
        print("inf," + ','.join('{:0.0f}'.format(i.numpy()) for i in _inf__mask))
        print("nan," + ','.join('{:0.0f}'.format(i.numpy()) for i in _nan__mask))

    return _total__num, _larg__num, _zero__num, _inf__num, _nan__num, _omit__num, _larg__mask, _zero__mask, _inf__mask, _nan__mask, _omit__mask

#
# statistic: g a b information
g0_total__num, g0_larg__num, g0_zero__num, g0_inf__num, g0_nan__num, g0_omit__num, g0_larg__mask, g0_zero__mask, g0_inf__mask, g0_nan__mask, g0_omit__mask = calc_base_info(g0)
a0_total__num, a0_larg__num, a0_zero__num, a0_inf__num, a0_nan__num, a0_omit__num, a0_larg__mask, a0_zero__mask, a0_inf__mask, a0_nan__mask, a0_omit__mask = calc_base_info(a0)
b0_total__num, b0_larg__num, b0_zero__num, b0_inf__num, b0_nan__num, b0_omit__num, b0_larg__mask, b0_zero__mask, b0_inf__mask, b0_nan__mask, b0_omit__mask = calc_base_info(b0)

print(f"_INFO,total,large,small_{VAL_RELATIVE_SMALL},zero,inf,nan,max,min,file")
print(f"G,{g0_total__num},{g0_larg__num},{g0_omit__num},{g0_zero__num},{g0_inf__num},{g0_nan__num},-,-,{fn_g}")
print(f"A,{a0_total__num},{a0_larg__num},{a0_omit__num},{a0_zero__num},{a0_inf__num},{a0_nan__num},-,-,{fn_a}")
print(f"B,{b0_total__num},{b0_larg__num},{b0_omit__num},{b0_zero__num},{b0_inf__num},{b0_nan__num},-,-,{fn_b}")


# all: filter: all no inf/nan, g no small, g no zero
all_pick__mask = torch.mul(g0_larg__mask, all_finite__mask)
all_pick__num = calc_mask_num(all_pick__mask)
all_omit__mask = torch.add(torch.add(g0_omit__mask, a0_omit__mask), b0_omit__mask)
all_omit__mask = torch.where(all_omit__mask > 0.0, 1.0, all_omit__mask)
all_omit__num = calc_mask_num(all_omit__mask)

# all: filter: g zero but other non zero
all_g_zero_and_other_not__mask = torch.sub(g0_zero__mask, all_zero__mask)
all_g_zero_and_other_not__num = calc_mask_num(all_g_zero_and_other_not__mask)

if is_debug:
    print("all_nz_but_g_z," + ','.join('{:0.0f}'.format(i.numpy()) for i in all_g_zero_and_other_not__mask))


# all: shaping by cast
if dtype_adjust_to == "bf16":
    g0 = g0.type(torch.bfloat16).type(torch.float64)
    a0 = a0.type(torch.bfloat16).type(torch.float64)
    b0 = b0.type(torch.bfloat16).type(torch.float64)
elif dtype_adjust_to == "fp16":
    g0 = g0.type(torch.float16).type(torch.float64)
    a0 = a0.type(torch.float16).type(torch.float64)
    b0 = b0.type(torch.float16).type(torch.float64)

#
# global functions
def calc_diff_by_topk(diff, topk, topk_max_limit):
    _topk = min(topk, topk_max_limit, diff.numel())
    if _topk != topk:
        return torch.tensor(0.0), None, 0
    _topk = torch.tensor(_topk).to(torch.int)
    _diff_topk, _diff_topk_arg = diff.flatten().topk(_topk)
    _diff_topk_avg = _diff_topk.mean()
    return _diff_topk_avg, _diff_topk_arg, _topk.numpy()

def calc_diff_by_arg(diff, arg):
    if arg is None:
        return torch.tensor(0.0), 0
    _diff_avg = diff.flatten()[arg].mean()
    return _diff_avg, arg.numel()

#
# measure zero: abs diff avg of g zero and other non zero
def calc_adiff(A, B, pick_mask=None, pick_num=None, is_abs=False):
    assert(pick_mask is not None)
    assert(pick_num is not None)
    if pick_num == 0.0:
        return torch.tensor(0.0), torch.tensor(0.0)
    _adiff = torch.sub(A, B)
    if is_abs:
        _adiff = torch.abs(_adiff)
    _adiff = torch.where(pick_mask != 0.0, _adiff, 0.0)
    _adiff_avg = torch.div(torch.sum(_adiff), pick_num)
    return _adiff, _adiff_avg

ga_adiff_g_zero, ga_adiff_g_zero_avg = calc_adiff(g0, a0, all_g_zero_and_other_not__mask, all_g_zero_and_other_not__num, is_abs=True)
gb_adiff_g_zero, gb_adiff_g_zero_avg = calc_adiff(g0, b0, all_g_zero_and_other_not__mask, all_g_zero_and_other_not__num, is_abs=True)
## GA adiff for topk of GA
ga_adiff_g_zero_top_gad_k1__num = 8
ga_adiff_g_zero_top_gad_k1_avg, ga_adiff_g_zero_top_gad_k1__arg, ga_adiff_g_zero_top_gad_k1__num = calc_diff_by_topk(ga_adiff_g_zero, ga_adiff_g_zero_top_gad_k1__num, all_g_zero_and_other_not__num)
## GB adiff for topk of GA
gb_adiff_g_zero_top_gad_k1__arg = ga_adiff_g_zero_top_gad_k1__arg
gb_adiff_g_zero_top_gad_k1_avg, gb_adiff_g_zero_top_gad_k1__num = calc_diff_by_arg(gb_adiff_g_zero, gb_adiff_g_zero_top_gad_k1__arg)
## GB adiff for topk of GB
gb_adiff_g_zero_top_gbd_k1__num = 8
gb_adiff_g_zero_top_gbd_k1_avg, gb_adiff_g_zero_top_gbd_k1__arg, gb_adiff_g_zero_top_gbd_k1__num = calc_diff_by_topk(gb_adiff_g_zero, gb_adiff_g_zero_top_gbd_k1__num, all_g_zero_and_other_not__num)
## GA adiff for topk of GB
ga_adiff_g_zero_top_gbd_k1__arg = gb_adiff_g_zero_top_gbd_k1__arg
ga_adiff_g_zero_top_gbd_k1_avg, ga_adiff_g_zero_top_gbd_k1__num = calc_diff_by_arg(ga_adiff_g_zero, ga_adiff_g_zero_top_gbd_k1__arg)

print(f"_AD_G_ZERO,avg_{all_g_zero_and_other_not__num},avg_ga_top_{ga_adiff_g_zero_top_gad_k1__num},avg_gb_top_{gb_adiff_g_zero_top_gbd_k1__num}")
print("GA," + ','.join('{:0.10f}'.format(i.numpy()) for i in [ga_adiff_g_zero_avg, ga_adiff_g_zero_top_gad_k1_avg, ga_adiff_g_zero_top_gbd_k1_avg]))
print("GB," + ','.join('{:0.10f}'.format(i.numpy()) for i in [gb_adiff_g_zero_avg, gb_adiff_g_zero_top_gad_k1_avg, gb_adiff_g_zero_top_gbd_k1_avg]))

#
# measure omit: abs diff avg of omit
ga_adiff_omit, ga_adiff_omit_avg = calc_adiff(g0, a0, all_omit__mask, all_omit__num, is_abs=True)
gb_adiff_omit, gb_adiff_omit_avg = calc_adiff(g0, b0, all_omit__mask, all_omit__num, is_abs=True)
## GA adiff for topk of GA
ga_adiff_omit_top_gad_k1__num = 8
ga_adiff_omit_top_gad_k1_avg, ga_adiff_omit_top_gad_k1__arg, ga_adiff_omit_top_gad_k1__num = calc_diff_by_topk(ga_adiff_omit, ga_adiff_omit_top_gad_k1__num, all_omit__num)
## GB adiff for topk of GA
gb_adiff_omit_top_gad_k1__arg = ga_adiff_omit_top_gad_k1__arg
gb_adiff_omit_top_gad_k1_avg, gb_adiff_omit_top_gad_k1__num = calc_diff_by_arg(gb_adiff_omit, gb_adiff_omit_top_gad_k1__arg)
## GB adiff for topk of GB
gb_adiff_omit_top_gbd_k1__num = 8
gb_adiff_omit_top_gbd_k1_avg, gb_adiff_omit_top_gbd_k1__arg, gb_adiff_omit_top_gbd_k1__num = calc_diff_by_topk(gb_adiff_omit, gb_adiff_omit_top_gbd_k1__num, all_omit__num)
## GA adiff for topk of GB
ga_adiff_omit_top_gbd_k1__arg = gb_adiff_omit_top_gbd_k1__arg
ga_adiff_omit_top_gbd_k1_avg, ga_adiff_omit_top_gbd_k1__num = calc_diff_by_arg(ga_adiff_omit, ga_adiff_omit_top_gbd_k1__arg)
print(f"TODO, AD_SMALL, avg_hi, avg_lo")
print(f"_AD_SMALL,avg_{all_omit__num},avg_ga_top_{ga_adiff_omit_top_gad_k1__num},avg_gb_top_{gb_adiff_omit_top_gbd_k1__num}")
print("GA," + ','.join('{:0.10f}'.format(i.numpy()) for i in [ga_adiff_omit_avg, ga_adiff_omit_top_gad_k1_avg, ga_adiff_omit_top_gbd_k1_avg]))
print("GB," + ','.join('{:0.10f}'.format(i.numpy()) for i in [gb_adiff_omit_avg, gb_adiff_omit_top_gad_k1_avg, gb_adiff_omit_top_gbd_k1_avg]))

#
# measure picked: rdiff_avg: npu <= gpu
def calc_rdiff(A, B, pick_mask=None, pick_num=None, is_abs=False):
    assert(pick_mask is not None)
    assert(pick_num is not None)
    if pick_num == 0.0:
        return torch.tensor(0.0), torch.tensor(0.0)
    _sub_abs = torch.abs(torch.sub(A, B))
    _abs_add = torch.add(torch.abs(A), torch.abs(B))
    _abs_add = torch.where(_abs_add == 0.0, 1.0, _abs_add)
    _rdiff = torch.div(_sub_abs, _abs_add)
    _rdiff = torch.where(pick_mask != 0.0, _rdiff, 0.0)
    _rdiff_avg = torch.div(torch.sum(_rdiff), pick_num)
    return _rdiff, _rdiff_avg

## calc GA
ga_rdiff_picked, ga_rdiff_picked_avg = calc_rdiff(g0, a0, all_pick__mask, all_pick__num)
ga_rdiff_picked_hi__mask = torch.where(ga_rdiff_picked > ga_rdiff_picked_avg, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
ga_rdiff_picked_hi__mask = torch.mul(ga_rdiff_picked_hi__mask, all_pick__mask)
ga_rdiff_picked_hi__num = calc_mask_num(ga_rdiff_picked_hi__mask)
ga_rdiff_picked_hi, ga_rdiff_picked_hi_avg = calc_rdiff(g0, a0, ga_rdiff_picked_hi__mask, ga_rdiff_picked_hi__num)
ga_rdiff_picked_lo__mask = torch.where(ga_rdiff_picked <= ga_rdiff_picked_avg, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
ga_rdiff_picked_lo__mask = torch.mul(ga_rdiff_picked_lo__mask, all_pick__mask)
ga_rdiff_picked_lo__num = calc_mask_num(ga_rdiff_picked_lo__mask)
ga_rdiff_picked_lo, ga_rdiff_picked_lo_avg = calc_rdiff(g0, a0, ga_rdiff_picked_lo__mask, ga_rdiff_picked_lo__num)
print(f"_RD_PICK,avg_{all_pick__num},avg_hi_{ga_rdiff_picked_hi__num},avg_lo_{ga_rdiff_picked_lo__num}")
print("GA," + ','.join('{:0.8f}'.format(i.numpy()) for i in [ga_rdiff_picked_avg, ga_rdiff_picked_hi_avg, ga_rdiff_picked_lo_avg]))

## calc GB XXX using GA's mask as base
gb_rdiff_picked, gb_rdiff_picked_avg = calc_rdiff(g0, b0, all_pick__mask, all_pick__num)
gb_rdiff_picked_hi__mask = ga_rdiff_picked_hi__mask
gb_rdiff_picked_hi__num = ga_rdiff_picked_hi__num
gb_rdiff_picked_hi, gb_rdiff_picked_hi_avg = calc_rdiff(g0, b0, gb_rdiff_picked_hi__mask, gb_rdiff_picked_hi__num)
gb_rdiff_picked_lo__mask = ga_rdiff_picked_lo__mask
gb_rdiff_picked_lo__num = ga_rdiff_picked_lo__num
gb_rdiff_picked_lo, gb_rdiff_picked_lo_avg = calc_rdiff(g0, b0, gb_rdiff_picked_lo__mask, gb_rdiff_picked_lo__num)
print("GB," + ','.join('{:0.8f}'.format(i.numpy()) for i in [gb_rdiff_picked_avg, gb_rdiff_picked_hi_avg, gb_rdiff_picked_lo_avg]))

#
# measure picked: rdiff_topk: npu <= gpu
## GA rdiff for topk of GA
ga_rdiff_picked_top_gad_k1__num = 1
ga_rdiff_picked_top_gad_k2__num = 8
ga_rdiff_picked_top_gad_k3__num = 128
ga_rdiff_picked_top_gad_k4__num = 1024
ga_rdiff_picked_top_gad_k1_avg, ga_rdiff_picked_top_gad_k1__arg, ga_rdiff_picked_top_gad_k1__num = calc_diff_by_topk(ga_rdiff_picked, ga_rdiff_picked_top_gad_k1__num, all_pick__num)
ga_rdiff_picked_top_gad_k2_avg, ga_rdiff_picked_top_gad_k2__arg, ga_rdiff_picked_top_gad_k2__num = calc_diff_by_topk(ga_rdiff_picked, ga_rdiff_picked_top_gad_k2__num, all_pick__num)
ga_rdiff_picked_top_gad_k3_avg, ga_rdiff_picked_top_gad_k3__arg, ga_rdiff_picked_top_gad_k3__num = calc_diff_by_topk(ga_rdiff_picked, ga_rdiff_picked_top_gad_k3__num, all_pick__num)
ga_rdiff_picked_top_gad_k4_avg, ga_rdiff_picked_top_gad_k4__arg, ga_rdiff_picked_top_gad_k4__num = calc_diff_by_topk(ga_rdiff_picked, ga_rdiff_picked_top_gad_k4__num, all_pick__num)
## GB rdiff for topk of GA
gb_rdiff_picked_top_gad_k1__arg = ga_rdiff_picked_top_gad_k1__arg
gb_rdiff_picked_top_gad_k2__arg = ga_rdiff_picked_top_gad_k2__arg
gb_rdiff_picked_top_gad_k3__arg = ga_rdiff_picked_top_gad_k3__arg
gb_rdiff_picked_top_gad_k4__arg = ga_rdiff_picked_top_gad_k4__arg
gb_rdiff_picked_top_gad_k1_avg, gb_rdiff_picked_top_gad_k1__num = calc_diff_by_arg(gb_rdiff_picked, gb_rdiff_picked_top_gad_k1__arg)
gb_rdiff_picked_top_gad_k2_avg, gb_rdiff_picked_top_gad_k2__num = calc_diff_by_arg(gb_rdiff_picked, gb_rdiff_picked_top_gad_k2__arg)
gb_rdiff_picked_top_gad_k3_avg, gb_rdiff_picked_top_gad_k3__num = calc_diff_by_arg(gb_rdiff_picked, gb_rdiff_picked_top_gad_k3__arg)
gb_rdiff_picked_top_gad_k4_avg, gb_rdiff_picked_top_gad_k4__num = calc_diff_by_arg(gb_rdiff_picked, gb_rdiff_picked_top_gad_k4__arg)

print(f"_RD_PICK_GA_TOP,avg_{ga_rdiff_picked_top_gad_k1__num},avg_{ga_rdiff_picked_top_gad_k2__num},avg_{ga_rdiff_picked_top_gad_k3__num},avg_{ga_rdiff_picked_top_gad_k4__num}")
print("GA," + ','.join('{:0.8f}'.format(i.numpy()) for i in [ga_rdiff_picked_top_gad_k1_avg, ga_rdiff_picked_top_gad_k2_avg, ga_rdiff_picked_top_gad_k3_avg, ga_rdiff_picked_top_gad_k4_avg]))
print("GB," + ','.join('{:0.8f}'.format(i.numpy()) for i in [gb_rdiff_picked_top_gad_k1_avg, gb_rdiff_picked_top_gad_k2_avg, gb_rdiff_picked_top_gad_k3_avg, gb_rdiff_picked_top_gad_k4_avg]))

## GB rdiff for topk of GB
gb_rdiff_picked_top_gbd_k1__num = 1
gb_rdiff_picked_top_gbd_k2__num = 8
gb_rdiff_picked_top_gbd_k3__num = 128
gb_rdiff_picked_top_gbd_k4__num = 1024
gb_rdiff_picked_top_gbd_k1_avg, gb_rdiff_picked_top_gbd_k1__arg, gb_rdiff_picked_top_gbd_k1__num = calc_diff_by_topk(gb_rdiff_picked, gb_rdiff_picked_top_gbd_k1__num, all_pick__num)
gb_rdiff_picked_top_gbd_k2_avg, gb_rdiff_picked_top_gbd_k2__arg, gb_rdiff_picked_top_gbd_k2__num = calc_diff_by_topk(gb_rdiff_picked, gb_rdiff_picked_top_gbd_k2__num, all_pick__num)
gb_rdiff_picked_top_gbd_k3_avg, gb_rdiff_picked_top_gbd_k3__arg, gb_rdiff_picked_top_gbd_k3__num = calc_diff_by_topk(gb_rdiff_picked, gb_rdiff_picked_top_gbd_k3__num, all_pick__num)
gb_rdiff_picked_top_gbd_k4_avg, gb_rdiff_picked_top_gbd_k4__arg, gb_rdiff_picked_top_gbd_k4__num = calc_diff_by_topk(gb_rdiff_picked, gb_rdiff_picked_top_gbd_k4__num, all_pick__num)
## GA rdiff for topk of GB
ga_rdiff_picked_top_gbd_k1__arg = gb_rdiff_picked_top_gbd_k1__arg
ga_rdiff_picked_top_gbd_k2__arg = gb_rdiff_picked_top_gbd_k2__arg
ga_rdiff_picked_top_gbd_k3__arg = gb_rdiff_picked_top_gbd_k3__arg
ga_rdiff_picked_top_gbd_k4__arg = gb_rdiff_picked_top_gbd_k4__arg
ga_rdiff_picked_top_gbd_k1_avg, ga_rdiff_picked_top_gbd_k1__num = calc_diff_by_arg(ga_rdiff_picked, ga_rdiff_picked_top_gbd_k1__arg)
ga_rdiff_picked_top_gbd_k2_avg, ga_rdiff_picked_top_gbd_k2__num = calc_diff_by_arg(ga_rdiff_picked, ga_rdiff_picked_top_gbd_k2__arg)
ga_rdiff_picked_top_gbd_k3_avg, ga_rdiff_picked_top_gbd_k3__num = calc_diff_by_arg(ga_rdiff_picked, ga_rdiff_picked_top_gbd_k3__arg)
ga_rdiff_picked_top_gbd_k4_avg, ga_rdiff_picked_top_gbd_k4__num = calc_diff_by_arg(ga_rdiff_picked, ga_rdiff_picked_top_gbd_k4__arg)

print(f"_RD_PICK_GB_TOP,avg_{ga_rdiff_picked_top_gbd_k1__num},avg_{ga_rdiff_picked_top_gbd_k2__num},avg_{ga_rdiff_picked_top_gbd_k3__num},avg_{ga_rdiff_picked_top_gbd_k4__num}")
print("GA," + ','.join('{:0.8f}'.format(i.numpy()) for i in [ga_rdiff_picked_top_gbd_k1_avg, ga_rdiff_picked_top_gbd_k2_avg, ga_rdiff_picked_top_gbd_k3_avg, ga_rdiff_picked_top_gbd_k4_avg]))
print("GB," + ','.join('{:0.8f}'.format(i.numpy()) for i in [gb_rdiff_picked_top_gbd_k1_avg, gb_rdiff_picked_top_gbd_k2_avg, gb_rdiff_picked_top_gbd_k3_avg, gb_rdiff_picked_top_gbd_k4_avg]))
