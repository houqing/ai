
if [ "$1" == "" ]; then is_do_parse=0; else is_do_parse=1; fi

#NAME="Loss/sa__summarize_target_assignment/"
#NAME="sa_predict/FeatureExtractor/resnet_v1_50/resnet_v1_50/conv1/BatchNorm"
NAME="sa_predict/FeatureExtractor/resnet_v1_50/resnet_v1_50/"
NAME="sa_predict/FeatureExtractor/resnet_v1_50/resnet_v1_50/block1"

GREP_OPT_PICKED="-e node,-,$NAME"

F_IN="graph.pbtxt--v5-op-info.csv"

F_DOT="a.dot"
F_SVG="a.svg"
F_PDF="a.pdf"

if [ "$is_do_parse" == "1" ]; then
echo "digraph g {" > "$F_DOT"
#NODE_PICKED="`cat "$F_IN" | grep $GREP_OPT_PICKED | grep -v -e "RestoreV2" -e "SaveV2" -e ",report_uninitialized_variables"`"
NODE_PICKED="`tail -n +2 "$F_IN" | grep $GREP_OPT_PICKED`"

GREP_OPT_PRE=`echo "$NODE_PICKED" | cut -d, -f 7 | grep -v -e "--" | sed -e 's/^/-e node,-,/g' -e 's/|/ -e node,-,/g' | tr '\r\n' ' ' | sed 's/  *$//g'`
NODE_PRE="`cat "$F_IN" | grep $GREP_OPT_PRE`"

GREP_OPT_POST=`echo "$NODE_PICKED" | cut -d, -f 3 | sed -e 's/^/-e [,|]/g' -e 's/$/[,|:]/g' | tr '\r\n' ' ' | sed 's/  *$//g'`
NODE_POST="`cat "$F_IN" | grep $GREP_OPT_POST`"

GREP_OPT_POST_PRE=`echo "$NODE_POST" | cut -d, -f 7 | grep -v -e "--" | sed -e 's/^/-e node,-,/g' -e 's/|/ -e node,-,/g' | tr '\r\n' ' ' | sed 's/  *$//g'`
NODE_POST_PRE="`cat "$F_IN" | grep $GREP_OPT_POST_PRE`"

NODE_RELATION="`( echo "$NODE_PICKED"; echo "$NODE_POST"; ) | sort | uniq`"
NODE_ALL="`( echo "$NODE_PRE"; echo "$NODE_PICKED"; echo "$NODE_POST"; echo "$NODE_POST_PRE"; ) | sort | uniq`"

echo "// node all def" | tee -a "$F_DOT"
echo "$NODE_ALL" | cut -d, -f 3,4,5,6 | sed -e 's/^\(.*\),\(.*\),\(.*\),\(.*\)$/"\1" [label="(\2)",tooltip="\1,\3,\4"];/g' >> "$F_DOT"


NODE_ALL_NAME="`echo "$NODE_ALL" | sort | uniq | cut -d, -f 3`"
echo "// node all group" | tee -a "$F_DOT"
for g in $NODE_ALL_NAME; do
	gg=`echo "$g" | sed "s/\// /g"`
	#if echo "$g" | grep -q "/"; then
	#	g1=`echo "$gg" | sed -e "s/ [^ ][^ ]*$//g"`
	#	g2=`echo "$gg" | sed -e "s/^.* \([^ ][^ ]*\)$/\1/g"`
	#else
	#	g1=""
	#	g2="$g"
	#fi
	_begin=""
	_end=""
	_name=""
	for i in $gg; do
		if [ "$i" == "" ]; then continue; fi
		if [ "$_name" == "" ]; then
			_name="$i"
		else
			_name="$_name/$i"
		fi
		_begin="${_begin}subgraph \"cluster_$_name\" { label=\"$i\"; "
		_end="; }$_end"
	done
	echo "$_begin \"$g\" $_end" >> "$F_DOT"
done

echo "// node all relation" | tee -a "$F_DOT"
echo "$NODE_RELATION" | cut -d, -f 3,7 | sed -e 's/^\(.*\),\(.*\)$/{ "\2" } -> "\1";/g' -e 's/|/", "/g' -e 's/{ "--" } -> //g' -e 's/:[0-9]*//g' >> "$F_DOT"
echo "}" >> "$F_DOT"
fi

#-Goverlap=scale 
DOT_OPT_G="-Grankdir=TB -Gmargin=2 -Gcolor=gray -Gfontcolor=gray -Gfontname=Arial -Gfontsize=6 -Glabeljust=l -Granksep=0.0 -Gnodesep=0.0 -Gpenwidth=0.3"
DOT_OPT_N="-Nshape=plain -Ncolor=dimgray -Nfontcolor=dimgray -Nfontname=Arial -Nfontsize=7"
DOT_OPT_E="-Earrowsize=0.5 -Ecolor=dimgray -Epenwidth=0.3 -Eweight=1"
DOT_OPT="-v $DOT_OPT_G $DOT_OPT_N $DOT_OPT_E"

dot $DOT_OPT -Tsvg "$F_DOT" > "$F_SVG"
#dot $DOT_OPT -Tpdf "$F_DOT" > "$F_PDF"


