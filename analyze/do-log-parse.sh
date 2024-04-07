#!/bin/bash

VER=6
TS=`date +%Y%m%d-%H%M%S`

FL="$@"
FO_ITER="out--$TS-iter.csv"
FO_BATCH="out--$TS-batch.csv"

sed_opt_gen_re_pre() {
       	_ID=0; GREP_OPT_FILTER_RE="";
       	SED_OPT1_N="case_v$VER"; SED_OPT1_K=".*"; SED_OPT1_V_=":"; 
       	SED_OPT2_N=""; SED_OPT2_K=".*"; SED_OPT2_V_=""; 
}
sed_opt_gen_re() {     	# <name> <group_total> <pattern_with_group>
       	if echo "$((_ID+1))" | grep -q "^[1-9]$"; then
       	       	SED_OPT1_N="${SED_OPT1_N},$1"
       	       	SED_OPT1_K="${SED_OPT1_K}$3.*"
       	       	for i in `seq $2`; do _ID=$((_ID+1)); SED_OPT1_V_="${SED_OPT1_V_},\\${_ID}"; done
       	else   	# XXX ASSERT($2 == 1)
       	       	SED_OPT2_N="${SED_OPT2_N},$1"
       	       	SED_OPT2_K="${SED_OPT2_K}$3.*"
       	       	for i in `seq $2`; do _ID=$((_ID+1)); _ID2=$((_ID-9)); SED_OPT2_V_="${SED_OPT2_V_},\\${_ID2}"; done
       	fi
       	# XXX ASSERT($_ID > 18)
       	if [ "=$4" == "=1" ]; then 
       	       	if [ "=$GREP_OPT_FILTER_RE" == "=" ]; then
       	       	       	GREP_OPT_FILTER_RE="$3"
       	       	       	SED_OPT_PREPROC="s#[\r\n]#\n#g;s#\(${GREP_OPT_FILTER_RE}\)#\n\1#g"
       	       	else
       	       	       	GREP_OPT_FILTER_RE="$GREP_OPT_FILTER_RE.*$3"
       	       	fi
       	fi
}
sed_opt_gen_re_post() {
       	SED_OPT1_K=`echo "$SED_OPT1_K" | tr "#" "@"`
       	SED_OPT2_K=`echo "$SED_OPT2_K" | tr "#" "@"`
       	SED_OPT_N="$SED_OPT1_N$SED_OPT2_N"
       	if [ "=$SED_OPT2_V_" == "=" ]; then
       	       	SED_OPT_PREFIX="s/#/@/g;s#${SED_OPT1_K}#"
       	       	SED_OPT_SUFFIX="${SED_OPT1_V_}#g"
       	else
       	       	SED_OPT_PREFIX="s/#/@/g;H;s#${SED_OPT2_K}#${SED_OPT2_V_}#g;x;s#${SED_OPT1_K}#"
       	       	SED_OPT_SUFFIX="${SED_OPT1_V_}#g;G;s#\n##g"
       	fi
       	SED_OPT_SUFFIX="${SED_OPT_SUFFIX};s/,nan\(,\|$\)/,-1.23456789\1/gi;s/False/0/gi;s/True/1/gi"
       	GREP_OPT_FILTER_RE=`echo "$GREP_OPT_FILTER_RE" | sed 's#^\.\*##g'`
       	echo "== FILTER: $GREP_OPT_FILTER_RE";
       	echo "== OPT1:"
       	echo "N: '$SED_OPT1_N'"; echo "K: '$SED_OPT1_K'"; echo "V: '$SED_OPT1_V_'";
       	echo "== OPT2:"
       	echo "N: '$SED_OPT2_N'"; echo "K: '$SED_OPT2_K'"; echo "V: '$SED_OPT2_V_'";
       	echo "== OPT:"
       	echo "HEAD: $SED_OPT_N"
       	echo "PRE : $SED_OPT_PREFIX"
       	echo "SUF : $SED_OPT_SUFFIX"
}


#### output for ITER

sed_opt_gen_re_pre
sed_opt_gen_re "lrank" 1 "local_rank: \([^,]\+\)"    1
sed_opt_gen_re "epoch" 1 " epoch: \([^,]\+\)"    1
sed_opt_gen_re "iter" 1 " step: \([^,]\+\)"    1
sed_opt_gen_re "loss" 1 " output is \[\([^]]\+\)"    1
sed_opt_gen_re "ov" 1 " overflow is \([^,]\+\)"
sed_opt_gen_re "ls" 1 " scale is \[\([^]]\+\)"
sed_opt_gen_re "gnorm" 1 " norm is \[\([^]]\+\)"    1
sed_opt_gen_re_post

echo "$SED_OPT_N" > "$FO_ITER"
for FI in $FL; do
       	if [ -e "$FI" ]; then echo "==== $FI"; else echo "==== skip [$FI]"; continue; fi
       	SED_OPT="${SED_OPT_PREFIX}${FI}${SED_OPT_SUFFIX}"

       	cat "$FI" \
       	       	| sed -e "$SED_OPT_PREPROC" \
       	       	| grep -a "$GREP_OPT_FILTER_RE" \
       	       	| sed -e "$SED_OPT" \
       	       	>> "$FO_ITER"
done


#### output for BATCH

sed_opt_gen_re_pre
sed_opt_gen_re "epoch" 1 "Train Epoch:[ \t]\+\([0-9]\+\)"    1
sed_opt_gen_re "iter" 1 "| Batch:[ \t]\+\([0-9]\+\)"    1
sed_opt_gen_re "iter_total" 1 "/[ \t]*\([0-9]\+\)"
sed_opt_gen_re "loss" 1 "| Average loss:[ \t]*\([^ \t]\+\)"    1
sed_opt_gen_re_post

echo "$SED_OPT_N" > "$FO_BATCH"
for FI in $FL; do
       	if [ -e "$FI" ]; then echo "==== $FI"; else echo "==== skip [$FI]"; continue; fi
       	SED_OPT="${SED_OPT_PREFIX}${FI}${SED_OPT_SUFFIX}"

       	cat "$FI" \
       	       	| sed -e "$SED_OPT_PREPROC" \
       	       	| grep -a "$GREP_OPT_FILTER_RE" \
       	       	| sed -e "$SED_OPT" \
       	       	>> "$FO_BATCH"
done
