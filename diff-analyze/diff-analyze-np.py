#!/usr/bin/env python3

import sys
import numpy as np


is_debug_data = False
is_debug_print = True
debug_print_num = 6
if is_debug_data:
    a = np.array([1,2,3,4,15,16,17,18], dtype=np.float16)
    b = np.array([11,22,33,44,0,66,77,32768], dtype=np.float16)
else:
    a_file = sys.argv[1]
    b_file = sys.argv[2]
    print("a: [", a_file)
    print("b: [", b_file)
    #a_file="/dev/urandom"
    #b_file="/dev/urandom"
    a = np.fromfile(a_file, dtype=np.float16, count=800)
    b = np.fromfile(b_file, dtype=np.float16, count=800)

a_u16 = a.view(np.uint16)
b_u16 = b.view(np.uint16)

if is_debug_print:
    print(a[0:debug_print_num])
    print(b[0:debug_print_num])
    print(a_u16[0:debug_print_num])
    print(b_u16[0:debug_print_num])

print("========")
print("total_num(A) \t: ", len(a))
print("total_num(B) \t: ", len(b))
print("count(INF(A)) \t: ", sum(np.isinf(a)))
print("count(INF(B)) \t: ", sum(np.isinf(b)))
print("count(NAN(A)) \t: ", sum(np.isnan(a)))
print("count(NAN(B)) \t: ", sum(np.isnan(b)))
print("--------")

diff_ba = b - a
diff_ab = a - b
print("(B-A) MAX|min \t: ", max(diff_ba), "\t|", min(diff_ba))
print("(A-B) min|MAX \t: ", min(diff_ab), "\t|", max(diff_ab))

b_nz_FIXME = np.copy(b)
b_nz_FIXME[b_nz_FIXME==0]=1
b_nz_FIXME.fill(1)
diff_ba_percent = diff_ba / b_nz_FIXME
print("max((B-A)/B) \t: ", max(diff_ba_percent))
print("sum(B-A)/N \t: ", sum(diff_ba)/len(diff_ba))
print("sum((B-A)/B)/N \t: ", sum(diff_ba_percent)/len(diff_ba))

a_nz_FIXME = np.copy(a)
a_nz_FIXME[a_nz_FIXME==0]=1
a_nz_FIXME.fill(1)
diff_ab_percent = diff_ab / a_nz_FIXME
print("max((A-B)/A) \t: ", max(diff_ab_percent))
print("sum(A-B)/N \t: ", sum(diff_ab)/len(diff_ab))
print("sum((A-B)/A)/N \t: ", sum(diff_ab_percent)/len(diff_ab))


if is_debug_print:
    print("A.bin:", end=" ")
    for i in a_u16[0:debug_print_num]:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
    print("B.bin:", end=" ")
    for i in b_u16[0:debug_print_num]:
        print(np.binary_repr(i, width=16), end=" ")
    print("")

t_xor = np.bitwise_xor(a_u16, b_u16)
t_xor[t_xor == 0]
if is_debug_print:
    print("T.xor___:", end=" ")
    for i in t_xor[0:debug_print_num]:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
mask_condition = t_xor != 0
t_xor_non_zero = np.extract(mask_condition, t_xor)
if is_debug_print:
    print("T.xor_nz:", end=" ")
    for i in t_xor_non_zero[0:debug_print_num]:
        print(np.binary_repr(i, width=16), end=" ")
    print("")
t_log2 = np.log2(t_xor_non_zero).astype(np.uint32)
if is_debug_print:
    print("log", t_log2[0:debug_print_num])
t_count = np.bincount(t_log2)
if is_debug_print:
    print("cnt", t_count)
print("count(A.exp==B.exp) \t: ", sum(t_count[0:10]))
print("count(A.mat==B.mat) [" + str(10) + "] \t: ", len(t_xor) - len(t_xor_non_zero))
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    if len(t_count) > i:
        print("count(A.mat==B.mat) [" + str(10 - 1 - i) + "] \t: ", t_count[i])

