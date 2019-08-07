#!/usr/bin/env python3

# -*- coding: utf-8 -*-

def my_parameter_exit():
    print("")
    print("Error! incorrect parameter")
    print('''
Usage: data-diff-analyzer.py <file_a> <file_b> <data_type> [threshold_diff] [threshold_diff_ratio]

Prameter:
    data_type            : int8, int16, int32, float16, float32
    threshold_diff       : the threshold for count the number of abs(A - B) 
                           greater than that threshold, default = 0.1
    threshold_diff_ratio : the threshold for count the number of 
                           abs((A - B) / A) and abs((B - A) / B) 
                           greater than that threshold, default = 0.1

Output:
    total_num(A)                      :  number of data_A
    total_num(B)                      :  number of data_B
    count(INF(A))                     :  number of infinite in data_A
    count(INF(B))                     :  number of infinite in data_B
    count(NAN(A))                     :  number of NAN in data_A
    count(NAN(B))                     :  number of NAN in data_B
    count(zero(A))                    :  number of zero in data_A
    count(zero(B))                    :  number of zero in data_B
    -------------------------
    count(abs(A-B) >= threshold_diff ):  number of abs(A-B) >= threshold_diff
    count(abs( (A-B)/A ) >= threshold_diff_ratio ): number of abs( (A-B)/A ) >=
                                                    threshold_diff_ratio
    count(abs( (B-A)/B ) >= threshold_diff_ratio ): number of abs( (B-A)/B ) >=
                                                    threshold_diff_ratio
    max(A-B) |max index               :  max value of (A-B) | index of max 
                                         value in (A-B)
    min(A-B) |min index               :  min value of (A-B) | index of min 
                                         value in (A-B)
    max(B-A) |max index               :  max value of (B-A) | index of max 
                                         value in (B-A)
    min(B-A) |min index               :  min value of (B-A) | index of min 
                                         value in (B-A)
    
    max((A-B)/A)|index                :  max value of (A-B)/A  | index of max 
                                         value in (A-B)/A
    max((B-A)/B)|index                :  max value of (B-A)/B  | index of max 
                                         value in (B-A)/B
    max(abs((A-B)/A))|index           :  max value of abs((A-B)/A)  | index of 
                                         max value in abs((A-B)/A)
    max(abs((B-A)/B))|index           :  max value of abs((B-A)/B)  | index of 
                                         max value in abs((B-A)/B)
    -------------------------
    sum(A-B)/N                        :  the average value of (A-B)
    sum((A-B)/A)/N                    :  the average value of (A-B)/A
    sum((B-A)/B)/N                    :  the average value of (B-A)/A
    sum(abs((A-B)/A))/N               :  the average value of abs((A-B)/A)
    sum(abs((B-A)/B))/N               :  the average value of abs((B-A)/A)
    count(A.exp==B.exp && A.sign==B.sign)   :  the number of data that A.exp==
                                               B.exp and A.sign==B.sign
    count(A.mat==B.mat) [N]           :  the number of data with the same high 
                                         N bits in the mantissa part ,when the 
                                         sign and exppnent is the same''')
    sys.exit()
def my_datatype_exit():
    print("")
    print("Error! incorrect data type")
    print("please enter the correct data type:float16|float32|int8|int16|int32")
    sys.exit()
def my_file_exit():
    print("")
    print("Error! file not exists! ")
    sys.exit()
def my_datalength_warn():
    print("")
    print("Warning! the length of two data is different")
def my_data_align_warn():
    print("")
    print("warning! The bytes of the data are not aligned")


import os
import sys
import numpy as np
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
'''

#the threshold of abs(a-b)
diff_ab_thres = 0.1
#the threshold of abs( (a-b)/a )
diff_ab_percent_thres = 0.1

is_debug_data = False
is_debug_print = False
#number of bits per data 
data_bit_width = 16
#the min byte length between two data set
data_byte_length = 0

data_type_list = ['int8', 'int16', 'int32', 'float16', 'float32']
if is_debug_data:
    debug_print_num = 10
    data_type = 'float16'
    a_data_test = [1,2,24,3,4,5,6,15,float('inf'),float('inf')]
    b_data_test = [0,22,22,33,44,55,66,float('nan'),11,float('inf')]
    a_byte_length = len(a_data_test)
    b_byte_length = len(b_data_test)
else:
    if(len(sys.argv) < 4 or len(sys.argv) > 6):
        my_parameter_exit()
    a_file = sys.argv[1]
    b_file = sys.argv[2]
    data_type = sys.argv[3]
    if(len(sys.argv) == 5):
        diff_ab_thres = float(sys.argv[4])
    if(len(sys.argv) == 6):
        diff_ab_thres = float(sys.argv[4])
        diff_ab_percent_thres = float(sys.argv[5])
#    a_file=r'D:\study\testdata\A1.bin'
#    b_file=r'D:\study\testdata\B1.bin'
#    data_type='float16'
    print("")
    print("a: [", a_file, "]")
    print("b: [", b_file, "]")
    print("data type:",data_type)
    #Determine if the file exists
    if not (os.path.isfile(a_file) and os.path.isfile(b_file)):
        my_file_exit()
    #Determine the byte length of the data
    a_byte_length = len(np.fromfile(a_file, dtype=np.int8, count=-1))
    b_byte_length = len(np.fromfile(b_file, dtype=np.int8, count=-1))
if data_type not in data_type_list:
    my_datatype_exit()
if data_type == 'float16':
    ddtype=np.float16
    data_bit_width = 16
if data_type == 'float32':
    ddtype=np.float32
    data_bit_width = 32
if data_type == 'int8':
    ddtype=np.int8
    data_bit_width = 8
if data_type == 'int16':
    ddtype=np.int16
    data_bit_width = 16
if data_type == 'int32':
    ddtype=np.int32
    data_bit_width = 32
if is_debug_data:
    a = np.array(a_data_test, dtype=ddtype)
    b = np.array(b_data_test, dtype=ddtype)
else:
    a = np.fromfile(a_file, dtype=ddtype, count=-1)
    b = np.fromfile(b_file, dtype=ddtype, count=-1)
        
if(len(a) != len(b)):
    my_datalength_warn()
    if(len(a) < len(b)):
        b = b[0:len(a)]
        data_byte_length = a_byte_length
    else:
        a = a[0:len(b)]
        data_byte_length = b_byte_length
if(data_bit_width == 16):
    if(data_byte_length % 2 != 0):
        my_data_align_warn()
        a = a[0:len(a)-1]
        b = b[0:len(b)-1]
if(data_bit_width == 32):
    if(data_byte_length % 4 != 0):
        my_data_align_warn()
        a = a[0:len(a)-1]
        b = b[0:len(b)-1]
if data_type == 'float16' or data_type == 'float32':
    a_uptype = a.astype(np.float64)
    b_uptype = b.astype(np.float64)
    if data_type == 'float16':
        a_type = a.view(np.uint16)
        b_type = b.view(np.uint16)
    else:
        a_type = a.view(np.uint32)
        b_type = b.view(np.uint32)        
else:
    a_type = a.view()
    b_type = b.view()
    a_uptype = a.astype(np.int64)
    b_uptype = b.astype(np.int64)

if is_debug_print:
    print(a[0:debug_print_num])
    print(b[0:debug_print_num])
    print(a_type[0:debug_print_num])
    print(b_type[0:debug_print_num])

print("=========================")
print("total_num(A)                    : ", len(a))
print("total_num(B)                    : ", len(b))

#result1
mask_condition_a = np.isinf(a)
a_not_inf = np.extract(mask_condition_a,a)
mask_condition_a = np.isnan(a)
a_not_nan = np.extract(mask_condition_a,a)
mask_condition_b = np.isinf(b)
b_not_inf = np.extract(mask_condition_b,b)
mask_condition_b = np.isnan(b)
b_not_nan = np.extract(mask_condition_b,b)
print("count(INF(A))                   : ", len(a_not_inf))
print("count(INF(B))                   : ", len(b_not_inf))
print("count(NAN(A))                   : ", len(a_not_nan))
print("count(NAN(B))                   : ", len(b_not_nan))

#result2
mask_condition_a_zero = np.equal(a,0)
a_zero = np.extract(mask_condition_a_zero,a)
mask_condition_b_zero = np.equal(b,0)
b_zero = np.extract(mask_condition_b_zero,b)
print("count(zero(A))                  : ", len(a_zero))
print("count(zero(B))                  : ", len(b_zero))
print("-------------------------")
np.seterr(divide='ignore', invalid='ignore')
diff_ab = a_uptype - b_uptype
np.seterr(divide='warn', invalid='warn')
mask_condition_diff_ab = np.isinf(diff_ab) + np.isnan(diff_ab) == 0
diff_ab_not_inf_nan = np.extract(mask_condition_diff_ab, diff_ab)
max_index_ab = np.where(diff_ab == max(diff_ab_not_inf_nan))
min_index_ab = np.where(diff_ab == min(diff_ab_not_inf_nan))

np.seterr(divide='ignore', invalid='ignore')
mask_condition_diff_ab_thres=abs(diff_ab) >= diff_ab_thres
np.seterr(divide='warn', invalid='warn')
mask_condition_diff_ab_thres = mask_condition_diff_ab_thres  &  mask_condition_diff_ab
diff_ab_over_thres = np.extract(mask_condition_diff_ab_thres, diff_ab)

np.seterr(divide='ignore', invalid='ignore')
diff_ba = b_uptype - a_uptype
np.seterr(divide='warn', invalid='warn')
mask_condition_diff_ba = np.isinf(diff_ba) + np.isnan(diff_ba) == 0
diff_ba_not_inf_nan = np.extract(mask_condition_diff_ba, diff_ba)
max_index_ba = np.where(diff_ba == max(diff_ba_not_inf_nan))
min_index_ba = np.where(diff_ba == min(diff_ba_not_inf_nan))
#result3
print("count(abs(A-B) >=", str(diff_ab_thres),")         : ", len(diff_ab_over_thres))

np.seterr(divide='ignore', invalid='ignore')
diff_ab_percent = diff_ab / a_uptype
np.seterr(divide='warn', invalid='warn')
mask_condition_a = np.isinf(diff_ab_percent) + np.isnan(diff_ab_percent) == 0
mask_condition_overflow_a = mask_condition_a_zero | mask_condition_a
error_index_a = np.where(mask_condition_overflow_a==0)
error_index_a = np.array(error_index_a)

mask_condition_a_not_zero = np.logical_not(mask_condition_a_zero)
mask_condition_effect = mask_condition_a_not_zero & mask_condition_a
#remove zero\inf
diff_ab_percent_effect = np.extract(mask_condition_effect, diff_ab_percent)
diff_ab_percent_effect_abs = abs(diff_ab_percent_effect)
diff_ab_percent_effect_abs_over_thres = np.where(diff_ab_percent_effect_abs >= diff_ab_percent_thres)
max_index_ab_percent = np.where(diff_ab_percent == max(diff_ab_percent_effect))
max_index_ab_percent_abs = np.where(abs(diff_ab_percent) == max(diff_ab_percent_effect_abs))

np.seterr(divide='ignore', invalid='ignore')
diff_ba_percent = diff_ba / b_uptype
np.seterr(divide='warn', invalid='warn')
mask_condition_b = np.isinf(diff_ba_percent) + np.isnan(diff_ba_percent) == 0
mask_condition_overflow_b = mask_condition_b_zero | mask_condition_b
error_index_b = np.where(mask_condition_overflow_b==0)
error_index_b = np.array(error_index_b)
mask_condition_b_not_zero = np.logical_not(mask_condition_b_zero)
mask_condition_effect = mask_condition_b_not_zero & mask_condition_b
#remove zero\inf
diff_ba_percent_effect = np.extract(mask_condition_effect,diff_ba_percent)
diff_ba_percent_effect_abs = abs(diff_ba_percent_effect)
diff_ba_percent_effect_abs_over_thres = np.where(diff_ba_percent_effect_abs >= diff_ab_percent_thres)
max_index_ba_percent = np.where(diff_ba_percent == max(diff_ba_percent_effect))
max_index_ba_percent_abs = np.where(abs(diff_ba_percent) == max(diff_ba_percent_effect_abs))

#result4
if( len(error_index_a[0]) > 0 ):
    print("Warning!  (A-B)/A data is nan or inf !  index = \n" ,error_index_a[0])
if( len(error_index_b[0]) > 0 ):
    print("Warning!  (B-A)/B data is nan or inf !  index = \n" ,error_index_b[0])
print("count(abs( (A-B)/A ) >=", str(diff_ab_percent_thres),")   : ", len(diff_ab_percent_effect_abs_over_thres[0]))
print("count(abs( (B-A)/B ) >=", str(diff_ab_percent_thres),")   : ", len(diff_ba_percent_effect_abs_over_thres[0]))
print("max(A-B) | max index            : ", max(diff_ab_not_inf_nan), " \t|", max_index_ab[0])
print("min(A-B) | min index            : ", min(diff_ab_not_inf_nan), " \t|", min_index_ab[0])
print("max(B-A) | max index            : ", max(diff_ba_not_inf_nan), " \t|", max_index_ba[0])
print("min(B-A) | min index            : ", min(diff_ba_not_inf_nan), " \t|", min_index_ba[0])
print("max((A-B)/A) | max index        : ", max(diff_ab_percent_effect), " \t|", max_index_ab_percent[0])
print("max((B-A)/B) | max index        : ", max(diff_ba_percent_effect), " \t|", max_index_ba_percent[0])
print("max(abs((A-B)/A)) | max index   : ", max(diff_ab_percent_effect_abs), " \t|", max_index_ab_percent_abs[0])
print("max(abs((B-A)/B)) | max index   : ", max(diff_ba_percent_effect_abs), " \t|", max_index_ba_percent_abs[0])
print("-------------------------")
#result5
diff_ab_average = diff_ab_not_inf_nan / len(diff_ab_not_inf_nan)
print("sum(A-B)/N                      : ", sum(diff_ab_average))
#result6
diff_ab_percent_average = diff_ab_percent_effect / len(diff_ab_percent_effect)
diff_ba_percent_average = diff_ba_percent_effect / len(diff_ba_percent_effect)
diff_ab_percent_average_abs = diff_ab_percent_effect_abs / len(diff_ab_percent_effect_abs)
diff_ba_percent_average_abs = diff_ba_percent_effect_abs / len(diff_ba_percent_effect_abs)
print("sum((A-B)/A)/N                  : ", sum(diff_ab_percent_average))
print("sum((B-A)/B)/N                  : ", sum(diff_ba_percent_average))
print("sum(abs((A-B)/A))/N             : ", sum(diff_ab_percent_average_abs))
print("sum(abs((B-A)/B))/N             : ", sum(diff_ba_percent_average_abs))
print("-------------------------")
#result7&result8
t_xor = np.bitwise_xor(a_type, b_type)
#remove nan\inf
t_xor_not_inf_nan = np.extract(mask_condition_diff_ab, t_xor)

if is_debug_print:
    print("T_xor___:", end=" ")
    for i in t_xor_not_inf_nan[0:debug_print_num]:
        print(np.binary_repr(i, width=data_bit_width), end=" ")
    print("")

mask_condition = t_xor_not_inf_nan != 0
t_xor_non_zero = np.extract(mask_condition, t_xor_not_inf_nan)

if is_debug_print:
    print("T_xor_non_zero:", end=" ")
    for i in t_xor_non_zero[0:debug_print_num]:
        print(np.binary_repr(i, width=data_bit_width), end=" ")
    print("")

mask_condition = t_xor_not_inf_nan < 0
t_xor_low_zero = np.extract(mask_condition, t_xor_not_inf_nan)

if data_type =='int8':
    t_xor_non_zero = t_xor_non_zero.astype(np.uint8)
if data_type == 'int16' or data_type == 'float16':
    t_xor_non_zero = t_xor_non_zero.astype(np.uint16)
if data_type == 'int32' or data_type == 'float32':
    t_xor_non_zero = t_xor_non_zero.astype(np.uint32)

t_log2 = np.log2(t_xor_non_zero).astype(np.uint32)

if is_debug_print:
    print("log", t_log2[0:debug_print_num])
t_count = np.bincount(t_log2)
if is_debug_print:
    print("t_count", t_count)

highN = 0
if(data_type == 'int8'):
    highN = 7
if(data_type == 'int16'):
    highN = 15
if(data_type == 'int32'):
    highN = 31
if(data_type == 'float16'):
    highN = 10
if(data_type == 'float32'):
    highN = 23
if(data_type == 'int8' or data_type == 'int16' or data_type == 'int32'):
    print("count(A.sign==B.sign)                 : ", len(t_xor_not_inf_nan) - len(t_xor_low_zero))
else:
    print("count(A.exp==B.exp && A.sign==B.sign) : ", len(t_xor_not_inf_nan) - len(t_xor_non_zero) + sum(t_count[0:highN]))
print("count(A.mat==B.mat) [" + str(highN) + "]              : ", len(t_xor_not_inf_nan) - len(t_xor_non_zero))
for i in range(highN):
    if len(t_count) > i:
        print("count(A.mat==B.mat) [" + str(highN - 1 - i) + "]              : ", t_count[i])
    else:
        print("count(A.mat==B.mat) [" + str(highN - 1 - i) + "]              : ", 0)

