

total_d1=0
total_d0=0

while read DATA; do
DATA="1$DATA"
#echo data="$DATA"

D1=`echo "ibase=16;obase=2;$DATA" | bc | grep -o 1 | wc -l`
D0=`echo "ibase=16;obase=2;$DATA" | bc | grep -o 0 | wc -l`

((total_d1+=D1-1))
((total_d0+=D0))

done


total=$((total_d1+total_d0))
echo total=$total d1=$total_d1 d0=$total_d0
echo ratio=$((total_d1*10000/total)) | sed "s/\(..\)$/.\1/g"
