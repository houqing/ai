#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys
import numpy as np


def my_parameter_exit():
    print("")
    print("Error! incorrect parameter")
    print('''
Usage:[code_address] [data_a_address] [data_b_address]''')
    exit()

is_debug_print = False
is_debug = False

if(len(sys.argv) !=3):
    my_parameter_exit()
a_file = sys.argv[1]
b_file = sys.argv[2]
#a_file = r"a"
#b_file = r"b"
data_type = 'float16'
data_a = np.fromfile(a_file, dtype=data_type, count=-1)
data_b = np.fromfile(b_file, dtype=data_type, count=-1)

print("data_a num total:","\t",len(data_a),"\t")
print("data_b_num_total:","\t",len(data_b),"\t")

mask_condition = np.isinf(data_a)
temp = np.extract(mask_condition, data_a)
print("data_a num inf:","\t",len(temp),"\t")
mask_condition = np.isnan(data_a)
temp = np.extract(mask_condition, data_a)
print("data_a num nan:","\t",len(temp),"\t")
mask_condition = np.equal(data_a, 0)
temp = np.extract(mask_condition, data_a)
print("data_a num zero:","\t",len(temp),"\t")
mask_condition = np.isinf(data_b)
temp = np.extract(mask_condition, data_b)
print("data_b num inf:","\t",len(temp),"\t")
mask_condition = np.isnan(data_b)
temp = np.extract(mask_condition, data_b)
print("data_b num nan:","\t",len(temp),"\t")
mask_condition = np.equal(data_b, 0)
temp = np.extract(mask_condition, data_b)
print("data_b num zero:","\t",len(temp),"\t")

if is_debug is True:
    data_a = []
    data_b = []
    for i in range(0, 16):
        data_a.append(1 << i)
        data_b.append(0xf000)
    data_a = [0, 0, 0, 1, 1, 0xf000, 3, 0xfbff]
    data_b = [0, 0, 1, 0,10, 0x7000,30, 0x7bff] 

    data_a = np.array(data_a, np.uint16)
    data_b = np.array(data_b, np.uint16)
    data_a = data_a.view(np.float16)
    data_b = data_b.view(np.float16)
#remove inf | nan
#c = data_a - data_b
mask_condition = np.isinf(data_a) + np.isnan(data_a) + np.isinf(data_b) + np.isnan(data_b) == 0
a_legal = np.extract(mask_condition, data_a)
b_legal = np.extract(mask_condition, data_b)

a_temp = a_legal.view(np.uint16)
b_temp = b_legal.view(np.uint16)
a_rm_sign = a_temp & 0x7fff
b_rm_sign = b_temp & 0x7fff

t_or = a_rm_sign | b_rm_sign

#not (a==0 and b==0)
mask_condition = t_or ==0

t_or_fill_zero = t_or + mask_condition
t_or_fill_zero = t_or_fill_zero.view(np.uint16)
ab_high_bit_fill_zero = np.log2(t_or_fill_zero).astype(np.uint32)
ab_high_bit = (ab_high_bit_fill_zero + 1) * (~ mask_condition)

diff_ab = np.abs(a_legal - b_legal)
#todo overflow! if overflow diff_ab is +inf or -inf
mask_condition_not_overflow = [np.isinf(diff_ab) == 0]
diff_ab_fill_overflow = np.select(mask_condition_not_overflow, [diff_ab], default=-65504)
diff_ab_fill_overflow = diff_ab_fill_overflow.astype(np.float16)
mask_condition_zero = diff_ab == 0

diff_ab_fill_zero = diff_ab_fill_overflow + mask_condition_zero
diff_ab_uint16 = diff_ab_fill_zero.view(np.uint16)

diff_ab_high_bit_fill_zero = np.log2(diff_ab_uint16).astype(np.uint32)

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
    for i in diff_ab_uint16:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    temp = a_legal.view(np.uint16)
    for i in temp:
        print(np.binary_repr(i, width=16), end=" ")
    print("")

print("")
print("bit num", end="\t")
for i in range(17):
    print(i, end="\t")
print("")
print("----------------")

for i in range(17):
    high_bitoff_ab_i = np.zeros(16)
    mask_condition =  ab_high_bit == 16 - i
    
    ab_high_bit_i_diff_ab_high_bit = np.extract(mask_condition, diff_ab_high_bit)
    
    t_count = np.bincount(ab_high_bit_i_diff_ab_high_bit)
    
    print("%3d" % (16 - i), ":",end="\t")
    for j in range(17):
        if j <len(t_count):
            print(t_count[j],end="\t")
        else:
            print(".",end="\t")
    
    print("")
    
