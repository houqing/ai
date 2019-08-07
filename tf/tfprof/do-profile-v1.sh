#!/bin/bash

OP="$1"
F="$2"

usage_exit () {
	echo "Usage: $0 <generate|parse-scope|parse-graph> <FILE>"
	exit "$1"
}

case $OP in
	generate|gen)	OP=gen ;;
	parse-scope)	OP=parse-scope ;;
	parse-graph)	OP=parse-graph ;;
	*) usage_exit 1 ;;
esac

if [ -f "$F" ]; then
	echo "Parsing file [$F]"
else
	echo "Error: File not found [$F]"; echo
	usage_exit 1
fi

SCOPE_SELECT="bytes,peak_bytes,residual_bytes,output_bytes,micros,accelerator_micros,cpu_micros,params,float_ops,occurrence,op_types,input_shapes"
SCOPE_OUT_START_LINE=21
OP_SELECT="bytes,peak_bytes,residual_bytes,output_bytes,micros,accelerator_micros,cpu_micros,params,float_ops,occurrence,op_types"
OP_OUT_START_LINE=20

# graph, scope, code, op
if [ "$OP" == "gen" ]; then
	set -x
	profiler graph --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT | tail -n +$SCOPE_OUT_START_LINE > "$F--graph"
	profiler scope --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT | tail -n +$SCOPE_OUT_START_LINE > "$F--scope"
	profiler code  --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT | tail -n +$SCOPE_OUT_START_LINE > "$F--code"
	profiler op    --profile_path="$F" --max_depth=1000 --select=$OP_SELECT | tail -n +$OP_OUT_START_LINE > "$F--op"

	profiler graph --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT --output=timeline:outfile="$F--graph.json"
	profiler scope --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT --output=timeline:outfile="$F--scope.json"
	profiler code  --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT --output=timeline:outfile="$F--code.json"
	# No timeline for op
	# profiler op    --profile_path="$F" --max_depth=1000 --select=$SCOPE_SELECT --output=timeline:outfile="$F--op.json"

	exit
fi


if [ "$OP" == "parse-scope" ]; then
	F_OUT="$F.csv"
	sed \
		-e "s/op count (run|defined)/op count run a\/op count run b|op count defined a\/op count defined b/g" \
		-e "s/ | /,/g" \
		-e "s/^\(  *\)/\1_TFProfRoot\//g" \
		-e "s/, )/, -)/g" \
		-e "s/(\([^,]*\), \(.*params\)/(\2/g" \
		-e "s/ (/,/g" \
		-e "s/)$//g" \
		-e "s/ params//g" \
		-e "s/ flops//g" \
		-e "s/,  */,/g" \
		\
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)k\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)m\([,\/|]\)/\1\20000\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)b\([,\/|]\)/\1\20000000\3/g" \
		-e "s/\([0-9][0-9]*\)B\([,\/|]\)/\1\2/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)KB\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)MB\([,\/|]\)/\1\20000\3/g" \
		-e "s/\([0-9][0-9]*\)us\([,\/|]\)/\1\2/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)ms\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)sec\([,\/|]\)/\1\20000\3/g" \
		\
		"$F" | python3 -u do-profile-scope-v1.py > $F_OUT

	exit
fi

if [ "$OP" == "parse-graph" ]; then
	F_OUT="$F.csv"
	sed \
		-e "s/op count (run|defined)/op count run a\/op count run b|op count defined a\/op count defined b/g" \
		-e "s/ | /,/g" \
		-e "s/^\(  *\)/\1_TFProfRoot\//g" \
		-e "s/, )/, -)/g" \
		-e "s/(\([^,]*\), \(.*params\)/(\2/g" \
		-e "s/ (/,/g" \
		-e "s/)$//g" \
		-e "s/ params//g" \
		-e "s/ flops//g" \
		-e "s/,  */,/g" \
		\
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)k\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)m\([,\/|]\)/\1\20000\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)b\([,\/|]\)/\1\20000000\3/g" \
		-e "s/\([0-9][0-9]*\)B\([,\/|]\)/\1\2/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)KB\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)MB\([,\/|]\)/\1\20000\3/g" \
		-e "s/\([0-9][0-9]*\)us\([,\/|]\)/\1\2/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)ms\([,\/|]\)/\1\20\3/g" \
		-e "s/\([0-9][0-9]*\)\.\([0-9][0-9]\)sec\([,\/|]\)/\1\20000\3/g" \
		\
		"$F" | python3 -u do-profile-graph-v1.py > $F_OUT

	exit
fi

exit
grep -v -e "  All" -e "  ArithmeticOptimizer" -e "  Assign" -e "  Cast" -e "  Const" profile_220--scope | head -2000 > profile_220--scope--trim
