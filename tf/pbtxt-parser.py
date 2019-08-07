
# houqing@(Turing Architecture and Design Dept, HS)

import sys
import re
from collections import Counter

import tensorflow as tf
from google.protobuf import text_format

VERSION='v5'

def usage_exit():
    print('Usage:', sys.argv[0], '<FILE>')
    exit(1)

if len(sys.argv) < 2:
    usage_exit()


my_file_in = sys.argv[1]
my_file_out_op_info = my_file_in + "--" + VERSION + "-op-info.csv"
my_file_out_op_count = my_file_in + "--" + VERSION + "-op-count.csv"
my_file_out_op_count_desc = my_file_in + "--" + VERSION + "-op-count-desc.csv"

my_op_info_csv_head = 'desc,info,node,op,dtype,shape,in-name,in-dtype,in-shape'
my_op_info_raw = []
my_op_info_csv = []
my_op_count_raw = []  # store op for parsing op count
my_op_count_csv_head = 'op,count'
my_op_count_csv = []
my_op_count_desc_name_list = []
my_op_count_desc_raw = []  # store op for parsing op count
my_op_count_desc_csv_head = 'desc,op,count'
my_op_count_desc_csv = []


my_special_op_list = ['SaveV2', 'RestoreV2', 'NoOp']


def do_load(file_in):
    # open file and get graph object
    with open(my_file_in) as f:
        txt = f.read()
    graph_def = text_format.Parse(txt, tf.GraphDef())
    return graph_def

def do_parse_op_info(node_list, desc, info):
    # parse op info
    for node in node_list:
        _dtype = []
        _shape = []
        _input = []
        if node.op not in my_special_op_list:   # FIXME input list for SaveV2 is too long and useless
            # parse op dtype info
            for _a in ['dtype', 'T', 'DstT', 'U']:  # 'T' and 'dtype' are duplicate for node{}; 'DstT' is for Cast
                if _a == 'T' and _dtype:
                    break
                if _a in node.attr:
                    _ls = str(node.attr.get(_a).list)
                    if _ls:
                        _t = re.findall('type: ([^ \n\r]*)', _ls)
                        if _t:
                            _s = ''
                            for _m in _t:
                                _s = _s + ' ' + _m if _s else _m
                            _dtype.append(_s.strip())
                    else:
                        _t = str(node.attr.get(_a)).lstrip('type: ')
                        _dtype.append(_t.strip())

            # parse op shape info
            for _a in ['_output_shapes', 'value']:  # '_output_shapes' and 'value' are duplicate for node{}
                if _a == 'value' and _shape:
                    break
                if _a in node.attr:
                    _l = str(node.attr.get(_a).list)
                    if _l:
                        for _l in node.attr.get(_a).list.shape:
                            _t = re.findall('size: ([0-9-]*)', str(_l))
                            if _t:
                                _s = ''
                                for _m in _t:
                                    _s = _s + ' ' + _m if _s else _m
                            else:
                                _s = 's'
                            _shape.append(_s.strip())
                    else:
                        _t = re.findall('size: ([0-9-]*)', str(node.attr.get(_a)))
                        if _t:
                            _s = ''
                            for _m in _t:
                                _s = _s + ' ' + _m if _s else _m
                        else:
                            _s = 's'
                        _shape.append(_s.strip())
            # generate input list
            _input = [i for i in node.input]
        else:   # skip parsing special op
            _dtype.append('#')
            _shape.append('#')
            _input.append('#')

        if not _dtype:  # not exist such field
            _dtype.append('-')
        if not _shape:  # not exist such field
            _shape.append('-')
        my_op_info_raw.append([desc, info, node.name, node.op, _dtype, _shape, _input])
        my_op_count_raw.append(node.op)
        if desc not in my_op_count_desc_name_list:
            my_op_count_desc_name_list.append(desc)
        my_op_count_desc_raw.append([desc, node.op])


def do_parse_op_count():
    # parse op count
    _result = Counter(my_op_count_raw).most_common()
    for k,v in _result:
        my_op_count_csv.append(str(k) + ',' + str(v))
    my_op_count_csv.sort()

def do_parse_op_count_desc():
    # parse op count according to desc
    _result = []
    for n in my_op_count_desc_name_list:
        _result.append([])

    for d, op in my_op_count_desc_raw:
        _result[my_op_count_desc_name_list.index(d)].append(op)

    for n in my_op_count_desc_name_list:
        _r = Counter(_result[my_op_count_desc_name_list.index(n)]).most_common()
        _r.sort()
        for k,v in _r:
            my_op_count_desc_csv.append(n + ',' + str(k) + ',' + str(v))


def do_save():
    # write to file
    with open(my_file_out_op_info, 'w') as f:
        f.write(my_op_info_csv_head + '\n')
        for node in my_op_info_raw:    # parse one node
            _input_list = []
            _input_dtype_list = []
            _input_shape_list = []
            _input_dtype_dup_list = []
            _input_shape_dup_list = []
            if node[6]:
                _input_list = node[6]
                for _input_node_name_raw in node[6]:   # parse all inputs in one node
                    _input_node_name = _input_node_name_raw.split(':')
                    if len(_input_node_name) == 1:  # for input name without ":"
                        _input_node_name.append('0')
                    elif len(_input_node_name) >= 3:    # ignore input format like "ParseSingleExample/Reshape_1:output:0"
                        _input_dtype_list.append('?')
                        _input_shape_list.append('?')
                        continue
                    elif len(_input_node_name) == 0:    # XXX might not happen for empty input list
                        _input_dtype_list.append('????????')
                        _input_shape_list.append('????????')
                        continue
                    _input_node_name[1] = int(_input_node_name[1])
                    _is_input_found = False
                    _input_duplicate_flag_left = ''
                    _input_duplicate_flag_right = ''
                    for _input_node in my_op_info_raw:    # search a input from raw info list
                        if _input_node_name[0] == _input_node[2]:
                            if _is_input_found: # found duplicate field
                                _input_duplicate_flag_left = '('
                                _input_duplicate_flag_right = ')'
                            _is_input_found = True
                            if _input_node[3] in my_special_op_list:    # skip special op from input node
                                _input_dtype_list.append('##')
                                _input_shape_list.append('##')
                                break

                            if _input_node[4]:
                                if len(_input_node[4]) == 1:
                                    _input_dtype = _input_node[4][0]
                                elif len(_input_node[4]) > _input_node_name[1]:
                                    _input_dtype = _input_node[4][_input_node_name[1]]
                                else:
                                    _input_dtype = _input_node[4][-1]
                                if _input_duplicate_flag_right:
                                    _input_dtype_list[-1] = _input_dtype_list[-1] + _input_duplicate_flag_left + _input_dtype + _input_duplicate_flag_right
                                else:
                                    _input_dtype_list.append(_input_dtype)
                            if _input_node[5]:
                                if len(_input_node[5]) == 1:
                                    _input_shape = _input_node[5][0]
                                elif len(_input_node[5]) > _input_node_name[1]:
                                    _input_shape = _input_node[5][_input_node_name[1]]
                                else:
                                    _input_shape = _input_node[5][-1]
                                if _input_duplicate_flag_right:
                                    _input_shape_list[-1] = _input_shape_list[-1] + _input_duplicate_flag_left + _input_shape + _input_duplicate_flag_right
                                else:
                                    _input_shape_list.append(_input_shape)
                    if not _is_input_found: # not found the input name
                        _input_dtype_list.append('---')
                        _input_shape_list.append('---')
            else:   # not exist such input
                _input_list.append('--')
                _input_dtype_list.append('--')
                _input_shape_list.append('--')
            _input_dtype_list.extend(_input_dtype_dup_list)
            _input_shape_list.extend(_input_shape_dup_list)
            f.write(','.join([*node[0:4], '|'.join(node[4]), '|'.join(node[5]), '|'.join(_input_list), '|'.join(_input_dtype_list), '|'.join(_input_shape_list)]) + '\n')

    with open(my_file_out_op_count, 'w') as f:
        f.write(my_op_count_csv_head + '\n')
        for i in my_op_count_csv:
            f.write(i + '\n')

    with open(my_file_out_op_count_desc, 'w') as f:
        f.write(my_op_count_desc_csv_head + '\n')
        for i in my_op_count_desc_csv:
            f.write(i + '\n')


# main work
graph_def = do_load(my_file_in)
do_parse_op_info(graph_def.node, 'node', '-')
for fn in graph_def.library.function:
    do_parse_op_info(fn.node_def, 'lib-fn', fn.signature.name)
do_parse_op_count()
do_parse_op_count_desc()
do_save()

