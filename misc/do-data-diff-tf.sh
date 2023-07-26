#!/bin/bash

TS=`date +%Y%m%d-%H%M%S`

DA="b6/ge_default_*_*/*/*/npy"
DB="a6/ge_default_*_*/*/*/npy"

SORT_OPT="$1"

for A in `find $DA -name "*.npy" | sort -V $SORT_OPT`; do
        F1=`basename $A | cut -d. -f1-4`
        F2=`basename $A | cut -d. -f6-`
        B=`ls $DB/$F1.*.$F2 2>/dev/null`

        if [ -f "$B" ]; then
                if grep -q diff_b16 "$B--diff.log" 2>/dev/null; then
                        echo "======== skip [$F1.*.$F2]"
                        #continue
                        python ~/bin/run-float-diff-v16--todo.py $A $B npy-npy nofig cd
                else
                        #python ~/bin/run-float-diff-v16--todo.py $A $B npy-npy-2048 nofig cd
                        python ~/bin/run-float-diff-v16--todo.py $A $B npy-npy nofig cd
                fi
        else
                echo "!!!!!!!! miss [$F1.*.$F2]" | tee -a missing-$TS.log
                continue
        fi
done
