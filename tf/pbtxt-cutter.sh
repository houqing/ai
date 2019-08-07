
F=graph.pbtxt
NAME="Loss/sa__summarize_target_assignment"
NAME="global_step"
NAME="bert/encoder"

cat "$F" | tr "\r\n" "#" | sed -e "s/node {/\nnode {/g" -e "s/library {/\nlibrary {/g" | grep "^node {" > "$F--nodes-raw"
grep "name: \"$NAME" "$F--nodes-raw" > "$F--picked-raw"

grep -o 'name: "[^"]*"' "$F--picked-raw" | sed -e 's/name:/input:/g' | sed 's/"$/[":]/g' > "$F--post-pattern"
grep -o 'input: "[^"]*"' "$F--picked-raw" | sed -e 's/input:/name:/g' -e 's/:[^ "]*"/"/g' > "$F--pre-pattern"

grep -f "$F--pre-pattern" "$F--nodes-raw" > "$F--pre-raw"
grep -f "$F--post-pattern" "$F--nodes-raw" > "$F--post-raw"

cat "$F--pre-raw" "$F--picked-raw" "$F--post-raw" | sort | uniq > "$F--out-raw"

sed "s/#/\n/g" "$F--out-raw" > "$F--out"
