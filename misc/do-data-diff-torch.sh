#!/bin/bash

TS=`date +%Y%m%d-%H%M%S`

D_B=path/to/data_a
D_A=path/to/data_b

SORT_OPT="$1"

for A in `find $D_A -name "*npy" | sort -V $SORT_OPT`; do
        F=`basename "$A"`
        B="$D_B/$F"
        if [ -f "$B" ]; then
                if grep -q diff_b16 "$B--diff.log" 2>/dev/null; then
                        echo "======== skip [$F]"
                        continue
                        python ~/bin/data-diff-analyzer.py $A $B npy-npy nofig cd
                else
                        #python ~/bin/data-diff-analyzer.py $A $B npy-npy-8192 nofig cd
                        python ~/bin/data-diff-analyzer.py $A $B npy-npy nofig
                fi
        else
                echo "!!!!!!!! miss [$F]" | tee -a missing-$TS.log
                continue
        fi
done
