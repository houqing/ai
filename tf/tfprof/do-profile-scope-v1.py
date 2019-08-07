
import sys
import csv


f_in = sys.stdin
f_out = sys.stdout

if True:
    reader = csv.reader(f_in)
    list_in = list(reader)

list_out_head = [list_in[0]]
list_out_body = []

# layer curr
print("do layer curr", file=sys.stderr)
list_in_body_01 = []
for e in list_in[1:]:   # remove list header
    layer_curr = e[0].count(" ") / 2
    e.insert(0, int(layer_curr))
    e[1] = e[1].strip()
    list_in_body_01.append(e)
list_out_head[0].insert(0, "layer_curr")
list_out_body = list_in_body_01

# is_leaf, layer max
print("do is leaf", file=sys.stderr)
list_in_body_01_partial = [e[:2] for e in list_in_body_01]
list_in_body_02 = []
for e in list_in_body_01:
    # layer max
    e_prefix = ''.join([e[1], "/"])
    list_related = [ ee for ee in list_in_body_01_partial if (e_prefix in ee[1] or e[1].strip() == ee[1].strip())]
    layer_curr_list = [n[0] for n in list_related]
    layer_max = max(layer_curr_list)
    e.insert(0, layer_max)
    # is_leaf
    is_leaf = "y" if e[0] == e[1] else "n"
    e.insert(0, is_leaf)

    list_in_body_02.append(e)
list_out_head[0].insert(0, "layer_max")
list_out_head[0].insert(0, "is_leaf")
list_out_body = list_in_body_02

# parse body
print("do body", file=sys.stderr)
list_layer_max = list_in_body_02[0][1]
list_in_body_03 = []
for e in list_in_body_02:
    e1_heads = e[:3]
    e2_layers = e[3].strip().split("/") + [e[2]] * list_layer_max
    e2_layers = e2_layers[:list_layer_max+1]
    e3_names = [e[3].strip()]
    e4_params = []
    for p in e[4:13]:
        e4_params += p.split("/")
    e5_op_types = [e[13]]
    e6_op_counts = []
    for t in e[14].split("|"):
        e6_op_counts += t.split("/")
    e7_shapes = [e[15]]

    list_in_body_03.append(e1_heads + e2_layers + e3_names + e4_params + e5_op_types + e6_op_counts + e7_shapes)
    if (len(list_in_body_03) % 1000) == 0:
        print(str(len(list_in_body_03) // 1000) + "k", file=sys.stderr, flush=True)

h1_heads = list_out_head[0][:3]
h2_layers = ["n"+str(i) for i in range(list_layer_max + 1)]
h3_names = [list_out_head[0][3]]
h4_params = []
for h in list_out_head[0][4:13]:
    h4_params += [h+" sum", h+" self"]
h5_op_types = [list_out_head[0][13]]
h6_op_counts = []
for t in list_out_head[0][14].split("|"):
    h6_op_counts += t.split("/")
h7_shapes = [list_out_head[0][15]]

list_out_head = [h1_heads + h2_layers + h3_names + h4_params + h5_op_types + h6_op_counts + h7_shapes]
list_out_body = list_in_body_03

list_out = list_out_head + list_out_body

writer = csv.writer(f_out)
for l in list_out:
    writer.writerow(l)

