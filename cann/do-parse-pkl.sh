#!/bin/bash                                                                                                             

MY_PREFIX="===="
D_MODEL="/npu/debug.*/solov2_ID3929_for_Pytorch"
D_LIB_PT="lib/python3.7/site-packages"
D_LIB_USR="/home/lwx1209626/mmcv"

R_MODEL=MDL
R_LIB_PT=LIB
R_LIB_USR=LIB

[ -f "$1" ] || exit 1

do_clean() {
       	if [ "=$2" == "=sort" ]; then
       	       	MY_WRAP="sort -V"
       	else
       	       	MY_WRAP="cat"
       	fi
       	MY_SKIP="$3"
       	cat "$1" \
       	       	| $MY_WRAP \
       	       	| sed \
       	       	-e "s#\"\], \[\"#\n${MY_PREFIX} #g" \
       	       	-e "s# \[\[\"#\n${MY_PREFIX} #g" \
       	       	| sed \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_LIB_PT}#${MY_PREFIX} ${R_LIB_PT}#g" \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_LIB_USR}#${MY_PREFIX} ${R_LIB_USR}#g" \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_MODEL}#${MY_PREFIX} ${R_MODEL}#g" \
       	       	-e 's#^\(\["[^]]*\][^]]*\]\).*#\1#g' \
       	       	| grep -v \
       	       	-e "${MY_SKIP}/torch/" \
       	       	-e "${MY_SKIP}modules/module.py.*_call_impl.*return forward_call(" \
       	       	-e "${MY_SKIP}modules/module.py.*_call_impl.*result = self.forward(" \
       	       	-e "__${MY_SKIP}nn/utils/clip_grad.py.*clip_grad_norm_.*p.grad.detach().mul_" \
       	       	-e "__${MY_SKIP}nn/utils/clip_grad.py.*clip_grad_norm_.*clip_coef = max_norm /" \
       	       	-e "${MY_SKIP}npu/profiler.py.*wrapper.*return func(" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/wrap_tensor.py.*tensor_op_template" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/wrap_tensor.py.*tensor_op_template.*return TensorOPTemplate(" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/hook_module.py.*__call__.*hook_result = hook(" \
       	       	> $1--stack-$2
}

do_clean "$1" cat
do_clean "$1" sort

grep -e '^\["[^"]*_\(input\|output\)' $1 | sort -V > $1--data
cut -d "]" -f 1,2 $1--data | sed -e "s/$/]/g" > $1--list

exit

       	cat "$1" \
       	       	| $MY_WRAP \
       	       	| sed \
       	       	-e "s#\"\], \[\"#\n${MY_PREFIX} #g" \
       	       	-e "s# \[\[\"#\n${MY_PREFIX} #g" \
       	       	| sed \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_LIB_PT}#${MY_PREFIX} ${R_LIB_PT}#g" \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_LIB_USR}#${MY_PREFIX} ${R_LIB_USR}#g" \
       	       	-e "s#^${MY_PREFIX} [^ ]*${D_MODEL}#${MY_PREFIX} ${R_MODEL}#g" \
       	       	-e 's#^\(\["[^]]*\][^]]*\]\).*#\1#g' \
       	       	| grep -v \
       	       	-e "${MY_SKIP}/torch/" \
       	       	-e "${MY_SKIP}modules/module.py.*_call_impl.*return forward_call(" \
       	       	-e "${MY_SKIP}modules/module.py.*_call_impl.*result = self.forward(" \
       	       	-e "__${MY_SKIP}nn/utils/clip_grad.py.*clip_grad_norm_.*p.grad.detach().mul_" \
       	       	-e "__${MY_SKIP}nn/utils/clip_grad.py.*clip_grad_norm_.*clip_coef = max_norm /" \
       	       	-e "${MY_SKIP}npu/profiler.py.*wrapper.*return func(" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/wrap_tensor.py.*tensor_op_template" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/wrap_tensor.py.*tensor_op_template.*return TensorOPTemplate(" \
       	       	-e "${MY_SKIP}ptdbg_ascend/hook_module/hook_module.py.*__call__.*hook_result = hook(" \
       	       	> $1--stack-$2
