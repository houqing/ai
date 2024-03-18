#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
import numpy as np


def my_parameter_exit():
    print("")
    print("Error! incorrect parameter")
    print('''
Usage:[code_address] [data_a_address] [data_b_address]''')

is_force = True
is_debug_print = False
is_debug = False

if(len(sys.argv) !=3):
    my_parameter_exit()
a_file = sys.argv[1]
b_file = sys.argv[2]
#a_file = r"aaaa.data"
#b_file = r"bbbb.data"
data_type = 'int16'
data_a = np.fromfile(a_file, dtype=data_type, count=-1)
data_b = np.fromfile(b_file, dtype=data_type, count=-1)

data_a = data_a.view(np.float16)
data_b = data_b.view(np.float16)

st_a_total = len(data_a)
st_b_total = len(data_b)

if is_force:
    total = min(st_a_total, st_b_total)
    data_a = data_a[:total]
    data_b = data_b[:total]

mask_condition = np.equal(data_a, 0)
st_a_zero = len(np.extract(mask_condition, data_a))
mask_condition_a = np.isinf(data_a)
st_a_inf = len(np.extract(mask_condition_a,data_a))
mask_condition_a = np.isnan(data_a)
st_a_nan = len(np.extract(mask_condition_a,data_a))

mask_condition = np.equal(data_b, 0)
st_b_zero = len(np.extract(mask_condition, data_b))
mask_condition_b = np.isinf(data_b)
st_b_inf = len(np.extract(mask_condition_b,data_b))
mask_condition_b = np.isnan(data_b)
st_b_nan = len(np.extract(mask_condition_b,data_b))

print("stat_A: total=" + str(st_a_total), "zero=" + str(st_a_zero), "inf=" + str(st_a_inf), "nan=" + str(st_a_nan))
print("stat_B: total=" + str(st_b_total), "zero=" + str(st_b_zero), "inf=" + str(st_b_inf), "nan=" + str(st_b_nan))


if is_debug is True:
    data_a = [0, 0x7000, 0xffff, 0xf000]
    data_b = [0, 0x7fff, 0x0001, 0xf000] 
    data_a = np.array(data_a, np.uint16)
    data_b = np.array(data_b, np.uint16)

a_temp = data_a.view(np.uint16)
b_temp = data_b.view(np.uint16)
a_rm_sign = a_temp & 0x7fff
b_rm_sign = b_temp & 0x7fff

t_or = a_rm_sign | b_rm_sign

#not (a==0 and b==0)
mask_condition = t_or ==0

t_or_fill_zero = t_or + mask_condition
t_or_fill_zero = t_or_fill_zero.view(np.uint16)
ab_high_bit_fill_zero = np.log2(t_or_fill_zero).astype(np.uint32)
ab_high_bit = (ab_high_bit_fill_zero + 1) * (~ mask_condition)

a_sign = a_temp & 0x8000
b_sign = b_temp & 0x8000
ab_sign_xor = a_sign ^ b_sign
mask_condition = ab_sign_xor == 0
sign_value = (-1) ** mask_condition
#temp = sign_value * b_rm_sign
diff_ab = np.abs(a_rm_sign + sign_value * b_rm_sign)
#data_a_int32 = np.array(data_a, np.int32)
#data_b_int32 = np.array(data_b, np.int32)
#diff_ab = np.abs(data_a_int32 - data_b_int32)

mask_condition_zero = diff_ab == 0

diff_ab_fill_zero = diff_ab + mask_condition_zero
#diff_ab_uint16 = diff_ab_fill_zero.view(np.uint16)

diff_ab_high_bit_fill_zero = np.log2(diff_ab_fill_zero).astype(np.uint32)

diff_ab_high_bit = (diff_ab_high_bit_fill_zero + 1) * (~ mask_condition_zero)



if is_debug_print is True:
    for i in a_temp:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    for i in b_temp:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    for i in b_rm_sign:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    for i in diff_ab_fill_zero:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    temp = data_a.view(np.uint16)
    for i in temp:
        print(np.binary_repr(i, width=16), end=" ")
    print("")

print("")
print("diff:", end="\t")
for i in range(17):
    print(i, end="\t")
print("")
print("----------------")

for i in range(17):
    #high_bitoff_ab_i = np.zeros(16)
    mask_condition =  ab_high_bit == 16 - i
    
    ab_high_bit_i_diff_ab_high_bit = np.extract(mask_condition, diff_ab_high_bit)
    
    t_count = np.bincount(ab_high_bit_i_diff_ab_high_bit)
    
    print("%3d" % (16 - i), ":",end="\t")
    for j in range(17):
        if j <len(t_count):
            print(t_count[j] if t_count[j] else ".",end="\t")
        else:
            print(".",end="\t")
    
    print("")
    
