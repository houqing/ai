#!/usr/bin/python

import re
import sys

my_ver = 'v8'

my_identity = 'houqing'


MY_EMPTY_STR = '#'
MY_GRAPH_TOOLTIP_LEN_LIMIT = 16300    # max=16384

# ~!@#$%^&_+-=;',.
def usage_exit(*args, **kwargs):
    print(*args, **kwargs)
    print('Usage:')
    print('  ' + sys.argv[0] + ' <FILE> [0-9]<,|;|+|-|#> <[+|-|!] [+|-|!]node_name_pattern|k=v ...>')
    print('  ' + sys.argv[0] + ' <FILE> [0-9]<,|;|+|-|#> = pattern_file')
    print('  ' + '  ' + '[0-9]*:', '(prefix) layer max; node basename level')
    print('  ' + '  ' + ',:', 'print matched')
    print('  ' + '  ' + ';:', 'print matched and relation')
    print('  ' + '  ' + '+:', 'graph top to bottom')
    print('  ' + '  ' + '-:', 'graph bottom to top')
    print('  ' + '  ' + '=:', 'shape info level')
    print('  ' + '  ' + '@:', 'node info with id')
    print('  ' + '  ' + '#:', 'debug')
    print('')
    print('Example:')
    print('  ' + sys.argv[0] + ' ge_onnx_ABC.pbtxt 2+@ = my_pattern_file')
    exit()

def MY_DEBUG_PRINT(*args, **kwargs):
    if my_mode_debug:
        print('#', file=sys.stderr, *args)

if len(sys.argv) <= 3:
    usage_exit('Error: param len', len(sys.argv))

my_fn_in = sys.argv[1]
my_mode = sys.argv[2]

my_mode_debug = False
my_mode_node_info_id = False
my_mode_print_matched_relation = False
my_mode_print_matched = False
my_mode_dot_graph_attr = None
my_mode_dot_graph_draw_shape_level = 0
my_mode_dot_graph_draw_node_basename_level = 0
my_mode_layer_max = -1

_t_str = ''
for _s in my_mode:
    if _s in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        if _t_str == '0':
            my_mode_dot_graph_draw_node_basename_level += 1
            _t_str = _s
        else:
            _t_str = _t_str + _s
    else:
        break
if _t_str:
    my_mode_layer_max = int(_t_str)

for _s in my_mode[len(_t_str)+my_mode_dot_graph_draw_node_basename_level:]:
    if _s in [',']:
        my_mode_print_matched = True
    elif _s in [';']:
        my_mode_print_matched_relation = True
    elif _s in ['+']:
        my_mode_dot_graph_attr = 'labelloc=b,rankdir=TB'
    elif _s in ['-']:
        my_mode_dot_graph_attr = 'labelloc=t,rankdir=BT'
    elif _s in ['=']:
        my_mode_dot_graph_draw_shape_level += 1
    elif _s in ['@']:
        my_mode_node_info_id = True
    elif _s in ['#']:
        my_mode_debug = True
    else:
        break


my_info = my_mode

MY_DEBUG_PRINT('# begin for:', my_info)

MY_NODE_PATTERN_TYPES = ['name', 'op', 'id', 'stream_id', 'tvm_blockdim']
# ~!@#$%^&_+-=;',.
MY_NODE_PATTERN_MODES = ['+', '-', '!', ',', ';']
MY_NODE_PATTERN_MODE_FILE = ['=']
MY_NODE_PATTERN_COMMENT = ['#']


MY_DEBUG_PRINT('parse pattern from pattern file')
my_remain_argv = sys.argv[3:]
my_pattern_file = None
my_node_grad_identity = '^$'
if my_remain_argv[0][0] in MY_NODE_PATTERN_MODE_FILE:
    if len(my_remain_argv[0]) > 1:
        my_node_grad_identity = my_remain_argv[0][1:]
    if len(my_remain_argv) == 2:
        my_pattern_file = my_remain_argv[1]
    else:
        usage_exit('Error: pattern:', my_remain_argv)

if my_pattern_file:
    with open(my_pattern_file, 'rb') as f:
        _t = f.read().decode()
        _t = _t.split('\n')
        my_pattern_raw_list = _t
        MY_DEBUG_PRINT('parsed pattern list from pattern file:', my_pattern_file)
        MY_DEBUG_PRINT(len(my_pattern_raw_list))
else:
    my_pattern_raw_list = sys.argv[3:]


MY_DEBUG_PRINT('parse node name pattern and compile into regex')

my_node_pattern_dict = {}
for k1 in MY_NODE_PATTERN_TYPES:
    my_node_pattern_dict[k1] = {}
    for k2 in MY_NODE_PATTERN_MODES:
        my_node_pattern_dict[k1][k2] = {}
        my_node_pattern_dict[k1][k2]['list'] = []
        my_node_pattern_dict[k1][k2]['re_list'] = []

# FIXME
#MY_MODE_OP_BUILDIN_BLACK_LIST = [ '-', "op=.*Variable", "op=.*Identity", '+' ]
MY_MODE_OP_BUILDIN_BLACK_LIST = []

my_pattern_list = MY_MODE_OP_BUILDIN_BLACK_LIST + my_pattern_raw_list
pattern_mode = '+'
_sel_pattern = ''
for _s in my_pattern_list:
    _s = _s.strip()
    if len(_s) == 1:
        if _s[0] in MY_NODE_PATTERN_MODES:
            pattern_mode = _s[0]
            _sel_pattern = ''
            continue
    elif len(_s) > 1:
        if _s[0] in MY_NODE_PATTERN_MODES:
            pattern_mode = _s[0]
            _sel_pattern = _s[1].strip()
        elif _s[0] in MY_NODE_PATTERN_COMMENT:
            _sel_pattern = ''
            continue
        else:
            _sel_pattern = _s
    else:
        continue

    if _sel_pattern:
        _sel = _sel_pattern.split('=')
        if len(_sel) == 2:
            _sel_k = _sel[0].strip()
            _sel_v = _sel[1].strip()
            if _sel_k in MY_NODE_PATTERN_TYPES:  # valid key for now
                _sel_pattern = _sel_v
            else:
                MY_DEBUG_PRINT('Error: bad key:', _sel_k)
                assert(False)
        elif len(_sel) == 1:
            _sel_k = 'name'
            _sel_v = _sel_pattern

        _sel_pattern_raw = _sel_pattern.lstrip('^').rstrip('$')
        _sel_pattern_re = '^' + _sel_pattern_raw + '$'
        _sel_pattern_list = my_node_pattern_dict[_sel_k][pattern_mode]['list']
        _sel_pattern_re_list = my_node_pattern_dict[_sel_k][pattern_mode]['re_list']
        if _sel_pattern_raw not in _sel_pattern_list:
            _sel_pattern_list.append(_sel_pattern_raw)
            _sel_pattern_re_list.append(re.compile(_sel_pattern_re))
        MY_DEBUG_PRINT('pattern    :', _sel_k, pattern_mode, my_node_pattern_dict[_sel_k][pattern_mode]['list'][-1])
        MY_DEBUG_PRINT('pattern_re :', _sel_k, pattern_mode, my_node_pattern_dict[_sel_k][pattern_mode]['re_list'][-1])

        if _sel_k == 'op' and _sel_pattern_raw[:3] != 'ge:':
            # add ge specific op name
            _sel_pattern_raw_alt = 'ge:' + _sel_pattern_raw
            _sel_pattern_re = '^' + _sel_pattern_raw_alt + '$'
            _sel_pattern_list = my_node_pattern_dict[_sel_k][pattern_mode]['list']
            _sel_pattern_re_list = my_node_pattern_dict[_sel_k][pattern_mode]['re_list']
            if _sel_pattern_raw_alt not in _sel_pattern_list:
                _sel_pattern_list.append(_sel_pattern_raw_alt)
                _sel_pattern_re_list.append(re.compile(_sel_pattern_re))
            MY_DEBUG_PRINT('pattern   =:', _sel_k, pattern_mode, my_node_pattern_dict[_sel_k][pattern_mode]['list'][-1])
            MY_DEBUG_PRINT('pattern_re=:', _sel_k, pattern_mode, my_node_pattern_dict[_sel_k][pattern_mode]['re_list'][-1])

'''
for k1 in MY_NODE_PATTERN_TYPES:
    for k2 in MY_NODE_PATTERN_MODES:
        if my_node_pattern_dict[k1][k2]['list']:
            MY_DEBUG_PRINT('pattern   :', k1, k2, my_node_pattern_dict[k1][k2]['list'])
            #MY_DEBUG_PRINT('pattern_re:', k1, k2, my_node_pattern_dict[k1][k2]['re_list'])
'''


def is_node_or_str_in_list(node_or_str, pattern_type, pattern_mode):
    assert(pattern_type in MY_NODE_PATTERN_TYPES)
    assert(pattern_mode in MY_NODE_PATTERN_MODES)
    if type(node_or_str) is str:
        _info = node_or_str
    else:
        _info = node_or_str[pattern_type]
    for p in my_node_pattern_dict[pattern_type][pattern_mode]['re_list']:
        if p.search(_info):
            return True
    return False

def is_node_selected(node):
    is_in_white_list = False
    is_selected = False
    for t in MY_NODE_PATTERN_TYPES:
        if is_node_or_str_in_list(node, t, '+'):
            is_in_white_list = True
            is_selected = True
            break
    if is_in_white_list:
        for t in MY_NODE_PATTERN_TYPES:
            for m in ['-', '!']:
                if is_node_or_str_in_list(node, t, m):
                    is_selected = False
                    break

    return is_selected

def is_node_blocked(node):
    is_blocked = False
    for t in MY_NODE_PATTERN_TYPES:
        if is_node_or_str_in_list(node, t, '!'):
            is_blocked = True
            break
    return is_blocked

def is_node_highlight(node):
    is_highlight = False
    for t in MY_NODE_PATTERN_TYPES:
        if is_node_or_str_in_list(node, t, ','):
            is_highlight = True
            break
    return is_highlight

def my_get_node_highlight_gid(node):
    _gid = 0
    for t in MY_NODE_PATTERN_TYPES:
        if is_node_or_str_in_list(node, t, ','):
            _gid = 1
            break

    for t in MY_NODE_PATTERN_TYPES:
        if is_node_or_str_in_list(node, t, ';'):
            _gid = 2 if _gid == 0 else 3
            break

    return _gid


MY_DEBUG_PRINT('read graph file and build node lines:', my_fn_in)
with open(my_fn_in, 'rb') as f:
    _t = f.read().decode()
    if 'ir_version: ' in _t:
        my_in_format = 'onnx'
    elif 'versions {' in _t and 'producer: ' in _t and 'min_consumer: ' in _t:
        my_in_format = 'tf'
    else:
        my_in_format = 'proto'
    MY_DEBUG_PRINT('detected graph format:', my_in_format)
    my_fn_out_dot = my_fn_in + '--' + my_info + '.dot'
    if my_in_format == 'onnx':
        _t = _t.split('\n')
        _t = ''.join([x for x in _t if x.startswith('  ') and not x.startswith('  name: ')])
        _t = re.sub(r'[\r\n]', r'', _t).strip()
        _t = re.sub(r'  node {', r'\nnode {', _t)
        my_in_lines = _t.split('\n')
    elif my_in_format == 'tf':
        _t = re.sub(r'[\r\n]', r'', _t).strip()
        _t = re.sub(r'node {', r'\nnode {', _t)
        _t = re.sub(r'library {', r'\nlibrary {', _t)
        _t = re.sub(r'versions {', r'\nversions {', _t)
        my_in_lines = re.findall(r'node {.*', _t)
    else:
        assert(False)
        _t = _t.split('\n')
        _t = ''.join([x for x in _t if x.startswith('    ') or x.startswith('  op {') or x.startswith('  }') or x.startswith('  attr {')])
        _t = re.sub(r'[^ ]  attr {.*', r'', _t)
        _t = re.sub(r'[\r\n]', r'', _t).strip()
        _t = re.sub(r'  op {', r'\nop {', _t)
        my_in_lines = _t.split('\n')


MY_DEBUG_PRINT('parse graph node lines:', len(my_in_lines))
# TODO connect between send and recv
MY_UNSUPPORTED_PATTERN = 'MY_UNSUPPORTED_PATTERN'
if my_in_format == 'onnx':
    # general node attr
    re_node_in_name_off_list = re.compile(r'    input: "([^ "]*)"')
    re_node_out_name_off_list = re.compile(r'    output: "([^ "]*)"')
    re_node_name = re.compile(r'    name: "([^ "]*)"')
    re_node_op = re.compile(r'    op_type: "([^ "]*)"')
    re_node_id = re.compile(r'      name: "id" * i: ([0-9-]*)')
    re_node_stream_id = re.compile(r'      name: "stream_id" * i: ([0-9-]*)')
    re_node_tvm_blockdim = re.compile(r'      name: "tvm_blockdim" * i: ([0-9-]*)')

    # other general node attr # TODO
    re_node_is_compiled_fusion_op = re.compile(r'      name: "_is_compiled_fusion_op" * i: ([0-9-]*)')
    re_node_is_n_batch_split = re.compile(r'      name: "_is_n_batch_split" * i: ([0-9-]*)')
    # _no_task: op is removed by fusion rule
    re_node_is_no_task = re.compile(r'      name: "_no_task" * i: ([0-9-]*)')
    re_node_fusion_scope = re.compile(r'      name: "fusion_scope" * i: ([0-9-]*)')
    re_node_input_i_list = re.compile(r'      name: "input_i" * ([^}]*)}')
    re_node_output_i_list = re.compile(r'      name: "output_i" * ([^}]*)}')
    re_node_continuous_input = re.compile(r'      name: "continuous_input" * i: ([0-9-]*)')
    re_node_continuous_input_alloc = re.compile(r'      name: "continuous_input_alloc" * i: ([0-9-]*)')
    re_node_continuous_output = re.compile(r'      name: "continuous_output" * i: ([0-9-]*)')

    # input/output
    re_node_param_off_section = re.compile(r'      name: "_input_name_value" * ([^}]*)}')
    re_node_param_name_section = re.compile(r'      name: "_input_name_key" * ([^}]*)}')

    # input:
    re_node_in_dtype_list = re.compile(r'      name: "input_desc_dtype:[^ ]*" * s: "([^ "]*)"')
    re_node_in_layout_list = re.compile(r'      name: "input_desc_layout:[^ ]*" * s: "([^ "]*)"')
    re_node_in_shape_list = re.compile(r'      name: "input_desc_shape:[^ ]*" * ([^}]*)}')
    re_node_in_dtype_orig_list = re.compile(r'      name: "input_desc_origin_dtype:[^ ]*" * s: "([^ "]*)"')
    re_node_in_layout_orig_list = re.compile(r'      name: "input_desc_origin_layout:[^ ]*" * s: "([^ "]*)"')
    re_node_in_shape_orig_list = re.compile(r'      name: "input_desc_origin_shape:[^ ]*" * ([^}]*)}')

    # output: not mandatory, just for storing
    re_node_out_dtype_list = re.compile(r'      name: "output_desc_dtype:[^ ]*" * s: "([^ "]*)"')
    re_node_out_layout_list = re.compile(r'      name: "output_desc_layout:[^ ]*" * s: "([^ "]*)"')
    re_node_out_shape_list = re.compile(r'      name: "output_desc_shape:[^ ]*" * ([^}]*)}')
    re_node_out_dtype_orig_list = re.compile(r'      name: "output_desc_origin_dtype:[^ ]*" * s: "([^ "]*)"')
    re_node_out_layout_orig_list = re.compile(r'      name: "output_desc_origin_layout:[^ ]*" * s: "([^ "]*)"')
    re_node_out_shape_orig_list = re.compile(r'      name: "output_desc_origin_shape:[^ ]*" * ([^}]*)}')

    # TODO: op specific attr

    # op matmulv2:
    #   ge pre: offset_x transpose_a transpose_b transpose_x1 transpose_x2

    # op batchmatmul:
    #   ge pre: adj_x adj_x1 adj_x2 adj_y

    # op layernorm:
    #   ge pre: begin_norm_axis begin_params_axis epsilon
    
    # op softmax:
    #   ge pre: axes

    # op reshape:
    #   ge pre: axis num_axes

    # op gatherv2:
    #   ge pre: batch_dims

    # op dropoutgenmask
    #   ge pre: seed seed2

    # op hcomallreduce:
    #   ge pre: fusion fusion_id

    # op reducesum
    #   ge pre: keep_dims

    re_attr_get_num_list = re.compile(r': ([0-9-]+)')
    re_attr_get_str_list = re.compile(r': "([^ "]*)"')
elif my_in_format == 'tf':
    # general node attr
    re_node_in_name_off_list = re.compile(r'  input: "([^ "]*)"')
    re_node_out_name_off_list = re.compile(MY_UNSUPPORTED_PATTERN)
    re_node_name = re.compile(r'  name: "([^ "]*)"')
    re_node_op = re.compile(r'  op: "([^ "]*)"')
    re_node_id = re.compile(r'  name: "([^ "]*)"')  # use name as id
    re_node_stream_id = re.compile(MY_UNSUPPORTED_PATTERN)
    re_node_tvm_blockdim = re.compile(MY_UNSUPPORTED_PATTERN)

    # input/output
    re_node_param_off_section = re.compile(MY_UNSUPPORTED_PATTERN)
    re_node_param_name_section = re.compile(MY_UNSUPPORTED_PATTERN)
else:
    re_node_in_name_off_list = re.compile(r'    input: "([^ "]*)"')
    assert(False)


MY_DEBUG_PRINT('define node name fix pattern')
# specific fix for tf1 node scope name
re_node_name_tf1_fix_01_bp_prefix = re.compile(r'^tower_[^/]*/gradients/tower_[^/]*/')    # '_r/_g/'
re_node_name_tf1_fix_02_bp_middle = re.compile(r'/tower_[^/]*/gradients/tower_[^/]*/')    # '/_r/_g/'
re_node_name_tf1_fix_03_bp_middle = re.compile(r'tower_[^/]*/gradients/tower_[^/]*/')    # '/_r/_g/'
re_node_name_tf1_fix_04_bp_prefix = re.compile(r'^tower_[^/]*/gradients/')    # '_r/_g/'
re_node_name_tf1_fix_05_bp_middle = re.compile(r'/tower_[^/]*/gradients/')    # '/_r/_g/'
re_node_name_tf1_fix_06_bp_middle = re.compile(r'tower_[^/]*/gradients/')    # '/_r/_g/'
#re_node_name_tf1_fix_07_bp_middle = re.compile(r'/gradients/')    # '/_g/'
re_node_name_tf1_fix_11_fp_prefix = re.compile(r'^tower_[^/]*/')    # '_r/'
re_node_name_tf1_fix_12_fp_middle = re.compile(r'/tower_[^/]*/')    # '/_r/'
re_node_name_tf1_fix_13_fp_middle = re.compile(r'tower_[^/]*/')    # '/_r/'
# specific fix for tf2 node scope name
re_node_name_tf2_fix_01_fp_prefix = re.compile(r'^While_body_while_body_[^/]*/while/')    # '_r/'
re_node_name_tf2_fix_02_fp_middle = re.compile(r'/While_body_while_body_[^/]*/while/')    # '/_r/'
re_node_name_tf2_fix_03_fp_middle = re.compile(r'While_body_while_body_[^/]*/while/')    # '/_r/'
re_node_name_tf2_fix_11_bp_prefix = re.compile(r'^While_body_while_body_[^/]*/gradient_tape/while/')    # '_r/_g/'
re_node_name_tf2_fix_12_bp_middle = re.compile(r'/While_body_while_body_[^/]*/gradient_tape/while/')    # '/_r/_g/'
re_node_name_tf2_fix_13_bp_middle = re.compile(r'While_body_while_body_[^/]*/gradient_tape/while/')    # '/_r/_g/'
re_node_name_tf2_fix_21_fb_prefix = re.compile(r'^While_body_while_body_[^/]*/')    # '_r/'
re_node_name_tf2_fix_22_fb_middle = re.compile(r'/While_body_while_body_[^/]*/')    # '/_r/'
re_node_name_tf2_fix_23_fb_middle = re.compile(r'While_body_while_body_[^/]*/')    # '/_r/'

re_node_name_my_fix_bp_middle = re.compile(my_node_grad_identity)   # '_g'
def node_name_strip(name):
    _name_strip = name
    _name_strip = re_node_name_tf1_fix_01_bp_prefix.sub('_r/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_02_bp_middle.sub('/_r/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_03_bp_middle.sub('/_r/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_04_bp_prefix.sub('_r/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_05_bp_middle.sub('/_r/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_06_bp_middle.sub('/_r/_g/', _name_strip)
    #_name_strip = re_node_name_tf1_fix_07_bp_middle.sub('/_g/', _name_strip)
    _name_strip = re_node_name_tf1_fix_11_fp_prefix.sub('_r/', _name_strip)
    _name_strip = re_node_name_tf1_fix_12_fp_middle.sub('/_r/', _name_strip)
    _name_strip = re_node_name_tf1_fix_13_fp_middle.sub('/_r/', _name_strip)

    _name_strip = re_node_name_tf2_fix_01_fp_prefix.sub('_r/', _name_strip)
    _name_strip = re_node_name_tf2_fix_02_fp_middle.sub('/_r/', _name_strip)
    _name_strip = re_node_name_tf2_fix_03_fp_middle.sub('/_r/', _name_strip)
    _name_strip = re_node_name_tf2_fix_11_bp_prefix.sub('_r/_g/', _name_strip)
    _name_strip = re_node_name_tf2_fix_12_bp_middle.sub('/_r/_g/', _name_strip)
    _name_strip = re_node_name_tf2_fix_13_bp_middle.sub('/_r/_g/', _name_strip)
    _name_strip = re_node_name_tf2_fix_21_fb_prefix.sub('_r/', _name_strip)
    _name_strip = re_node_name_tf2_fix_22_fb_middle.sub('/_r/', _name_strip)
    _name_strip = re_node_name_tf2_fix_23_fb_middle.sub('/_r/', _name_strip)

    _name_strip = re_node_name_my_fix_bp_middle.sub('_g', _name_strip)
    return _name_strip

my_nodes = {}
my_node_id_max_num = 0
my_node_stream_id_max_num = 0
for _line in my_in_lines:
    # parse: base
    _name = re_node_name.findall(_line)[0]
    _name_strip = node_name_strip(_name)
    _op = re_node_op.findall(_line)[0]
    _id = re_node_id.findall(_line)
    if _id:
        _id = _id[0]
        my_node_id_max_num = max(my_node_id_max_num, int(_id))
    else:
        _id = MY_EMPTY_STR
    _stream_id = re_node_stream_id.findall(_line)
    if _stream_id:
        _stream_id = _stream_id[0]
        my_node_stream_id_max_num = max(my_node_stream_id_max_num, int(_stream_id))
    else:
        _stream_id = MY_EMPTY_STR
    _tvm_blockdim = re_node_tvm_blockdim.findall(_line)
    _tvm_blockdim = _tvm_blockdim[0] if _tvm_blockdim else MY_EMPTY_STR

    _op_attr_list = []  # TODO

    # parse: optional param
    _param_off_section = re_node_param_off_section.findall(_line)
    _param_off_unsorted_list = re_attr_get_num_list.findall(_param_off_section[0]) if _param_off_section else []
    _param_name_section = re_node_param_name_section.findall(_line)
    _param_name_unsorted_list = re_attr_get_str_list.findall(_param_name_section[0]) if _param_name_section else []
    assert(len(_param_off_unsorted_list) == len(_param_name_unsorted_list))
    _param_optional_dict = {}
    for i in range(len(_param_off_unsorted_list)):
        _param_optional_dict[_param_off_unsorted_list[i]] = _param_name_unsorted_list[i]

    # parse: node input brief
    _in_name_off_list = re_node_in_name_off_list.findall(_line)
    _in_name_off_list_fix = [t if t else MY_EMPTY_STR + ':' + MY_EMPTY_STR for t in _in_name_off_list]
    _in_prev_node_out_name_off_dict = {}
    for i,t in enumerate(_in_name_off_list_fix):
        _in_prev_node_out_name_off_dict[str(i)] = [t.split(":")[0], t.split(":")[1]]

    # parse: node output brief
    _out_name_off_list = re_node_out_name_off_list.findall(_line)
    _out_name_off_list_fix = [t if t else MY_EMPTY_STR + ':' + MY_EMPTY_STR for t in _out_name_off_list]
    _out_self_node_out_name_off_dict = {}
    for i,t in enumerate(_out_name_off_list_fix):
        _out_self_node_out_name_off_dict[str(i)] = [t.split(":")[0], t.split(":")[1]]

    # parse: node input detail
    _in_dtype_list = re_node_in_dtype_list.findall(_line)
    _in_layout_list = re_node_in_layout_list.findall(_line)
    _in_shape_list = re_node_in_shape_list.findall(_line)
    _in_shape_list = [','.join(re_attr_get_num_list.findall(t)) for t in _in_shape_list]
    _in_shape_list = ['' + t + '' if t else MY_EMPTY_STR for t in _in_shape_list]
    _in_dtype_orig_list = re_node_in_dtype_orig_list.findall(_line)
    _in_layout_orig_list = re_node_in_layout_orig_list.findall(_line)
    _in_shape_orig_list = re_node_in_shape_orig_list.findall(_line)
    _in_shape_orig_list = [','.join(re_attr_get_num_list.findall(t)) for t in _in_shape_orig_list]
    _in_shape_orig_list = ['' + t + '' if t else MY_EMPTY_STR for t in _in_shape_orig_list]

    # parse: node output detail
    _out_dtype_list = re_node_out_dtype_list.findall(_line)
    _out_layout_list = re_node_out_layout_list.findall(_line)
    _out_shape_list = re_node_out_shape_list.findall(_line)
    _out_shape_list = [','.join(re_attr_get_num_list.findall(t)) for t in _out_shape_list]
    _out_shape_list = ['' + t + '' if t else MY_EMPTY_STR for t in _out_shape_list]
    _out_dtype_orig_list = re_node_out_dtype_orig_list.findall(_line)
    _out_layout_orig_list = re_node_out_layout_orig_list.findall(_line)
    _out_shape_orig_list = re_node_out_shape_orig_list.findall(_line)
    _out_shape_orig_list = [','.join(re_attr_get_num_list.findall(t)) for t in _out_shape_orig_list]
    _out_shape_orig_list = ['' + t + '' if t else MY_EMPTY_STR for t in _out_shape_orig_list]

    # buildup node input dict
    _node_in_data_dict = {}
    _node_in_ctrl_dict = {}
    _prev_node_data_out_list_dict = {}
    _prev_node_ctrl_out_list_dict = {}
    _node_in_info = ''
    _node_in_info_orig = ''
    for k in _in_prev_node_out_name_off_dict.keys():
        k_num = int(k)
        _node_in_v = {}
        _node_in_v['off'] = k
        _node_in_v['prev_node_out_name'] = _in_prev_node_out_name_off_dict[k][0]
        _node_in_v['prev_node_out_off'] = _in_prev_node_out_name_off_dict[k][1]
        if _in_prev_node_out_name_off_dict[k][1] != '-1':
            _node_in_v['param_name'] = _param_optional_dict[k] if k in _param_optional_dict else MY_EMPTY_STR
            _node_in_v['dtype'] = _in_dtype_list[k_num]
            _node_in_v['layout'] = _in_layout_list[k_num]
            _node_in_v['shape'] = _in_shape_list[k_num]
            _node_in_v['dtype_orig'] = _in_dtype_orig_list[k_num]
            _node_in_v['layout_orig'] = _in_layout_orig_list[k_num]
            _node_in_v['shape_orig'] = _in_shape_orig_list[k_num]
            _node_in = {}
            _node_in_data_dict[k] = _node_in_v
            _prev_node_data_out_list_dict[k] = _in_prev_node_out_name_off_dict[k]
            if _node_in_info:
                _node_in_info = _node_in_info + '|' + ':'.join([_node_in_v['dtype'], _node_in_v['layout'], _node_in_v['shape']])
                _node_in_info_orig = _node_in_info_orig + '|' + ':'.join([_node_in_v['dtype_orig'], _node_in_v['layout_orig'], _node_in_v['shape_orig']])
            else:
                _node_in_info = ':'.join([_node_in_v['dtype'], _node_in_v['layout'], _node_in_v['shape']])
                _node_in_info_orig = ':'.join([_node_in_v['dtype_orig'], _node_in_v['layout_orig'], _node_in_v['shape_orig']])
        else:
            _node_in_ctrl_dict[k] = _node_in_v
            _prev_node_ctrl_out_list_dict[k] = _in_prev_node_out_name_off_dict[k]

    # buildup node output dict
    _node_out_data_dict = {}
    _node_out_ctrl_dict = {}
    _node_out_info = ''
    _node_out_info_orig = ''
    for k in _out_self_node_out_name_off_dict.keys():
        k_num = int(k)
        _node_out_v = {}
        _node_out_v['off'] = k
        _node_out_v['self_node_out_name'] = _out_self_node_out_name_off_dict[k][0]
        _node_out_v['self_node_out_off'] = _out_self_node_out_name_off_dict[k][1]
        if _out_self_node_out_name_off_dict[k][1] != '-1':
            _node_out_v['dtype'] = _out_dtype_list[k_num]
            _node_out_v['layout'] = _out_layout_list[k_num]
            _node_out_v['shape'] = _out_shape_list[k_num]
            _node_out_v['dtype_orig'] = _out_dtype_orig_list[k_num]
            _node_out_v['layout_orig'] = _out_layout_orig_list[k_num]
            _node_out_v['shape_orig'] = _out_shape_orig_list[k_num]
            _node_out_data_dict[k] = _node_out_v
            if _node_out_info:
                _node_out_info = _node_out_info + '|' + ':'.join([_node_out_v['dtype'], _node_out_v['layout'], _node_out_v['shape']])
                _node_out_info_orig = _node_out_info_orig + '|' + ':'.join([_node_out_v['dtype'], _node_out_v['layout'], _node_out_v['shape']])
            else:
                _node_out_info = ':'.join([_node_out_v['dtype'], _node_out_v['layout'], _node_out_v['shape']])
                _node_out_info_orig = ':'.join([_node_out_v['dtype_orig'], _node_out_v['layout_orig'], _node_out_v['shape_orig']])
        else:
            _node_out_ctrl_dict[k] = _node_out_v

    # buildup node dict
    _node = {}
    _node['name'] = _name
    _node['name_strip'] = _name_strip
    _node['op'] = _op
    _node['op_reformat'] = _op.replace('ge:', '')
    _node['op_attr_list'] = _op_attr_list
    _node['id'] = _id
    _node['stream_id'] = _stream_id
    _node['tvm_blockdim'] = _tvm_blockdim
    _node['in_data_dict'] = _node_in_data_dict
    _node['in_ctrl_dict'] = _node_in_ctrl_dict
    _node['out_data_dict'] = _node_out_data_dict
    _node['out_ctrl_dict'] = _node_out_ctrl_dict
    _node['prev_node_data_out_dict'] = _prev_node_data_out_list_dict
    _node['prev_node_ctrl_out_dict'] = _prev_node_ctrl_out_list_dict
    _node['next_node_data_in_list_dict'] = {}   # output is one-to-many
    _node['next_node_ctrl_in_list_dict'] = {}   # output is one-to-many

    _node['info_in'] = _node_in_info
    _node['info_in_orig'] = _node_in_info_orig
    _node['info_out'] = _node_out_info
    _node['info_out_orig'] = _node_out_info_orig
    _node['uid'] = '.'.join([_node['stream_id'], _node['id']])

    my_nodes[_name] = _node

MY_DEBUG_PRINT('build node relations by going thru the current\'s previous nodes')
for _node in my_nodes.values():
    for k in _node['prev_node_data_out_dict'].keys():
        _prev_name_off = _node['prev_node_data_out_dict'][k]
        _prev_name = _prev_name_off[0]
        _prev_off = _prev_name_off[1]
        if _prev_name != MY_EMPTY_STR:
            _prev_node = my_nodes[_prev_name]
            _prev_next_name_off = _prev_node['next_node_data_in_list_dict']
            if _prev_off not in _prev_next_name_off:
                _prev_next_name_off[_prev_off] = []
            _prev_next_name_off[_prev_off].append([_node['name'], k])

    for k in _node['prev_node_ctrl_out_dict'].keys():
        _prev_name_off = _node['prev_node_ctrl_out_dict'][k]
        _prev_name = _prev_name_off[0]
        _prev_off = _prev_name_off[1]
        if _prev_name != MY_EMPTY_STR:
            _prev_node = my_nodes[_prev_name]
            _prev_next_name_off = _prev_node['next_node_ctrl_in_list_dict']
            if _prev_off not in _prev_next_name_off:
                _prev_next_name_off[_prev_off] = []
            _prev_next_name_off[_prev_off].append([_node['name'], k])


MY_DEBUG_PRINT('select nodes by pattern rules')
my_nodes_selected = {}
for _node in my_nodes.values():
    if is_node_selected(_node):
        my_nodes_selected[_node['name']] = _node
        if my_mode_print_matched or my_mode_print_matched_relation:
            print(_node['name'])
            if my_mode_print_matched_relation:
                for k in _node['prev_node_data_out_dict'].keys():
                    _t = _node['prev_node_data_out_dict'][k]
                    print(' ' + k + '<' + _t[1] + '\t' + _t[0])
                for k in _node['prev_node_ctrl_out_dict'].keys():
                    _t = _node['prev_node_ctrl_out_dict'][k]
                    print(' ' + k + '<' + _t[1] + '\t' + _t[0])
                for k in _node['next_node_data_in_list_dict'].keys():
                    _t_list = _node['next_node_data_in_list_dict'][k]
                    for _t in _t_list:
                        print(' ' + k + '>' + _t[1] + '\t' + _t[0])
                for k in _node['next_node_ctrl_in_list_dict'].keys():
                    _t_list = _node['next_node_ctrl_in_list_dict'][k]
                    for _t in _t_list:
                        print(' ' + k + '>' + _t[1] + '\t' + _t[0])

if not my_mode_dot_graph_attr:
    MY_DEBUG_PRINT('# end, skip dot graph build')
    exit(0)


def get_layer_list_by_node_name_strip(name_strip):
    _t = name_strip
    _t = _t.rstrip('/').split('/')
    if my_mode_dot_graph_draw_node_basename_level > 0:
        _t_a_list_raw = _t[:-my_mode_dot_graph_draw_node_basename_level]
        _t_b = '/'.join(_t[-my_mode_dot_graph_draw_node_basename_level:])
    else:
        _t_a_list_raw = _t
        _t_b = ''
    _t_real_max = my_mode_layer_max
    if my_mode_layer_max > 0:
        if '_g' in _t_a_list_raw:
            _t_real_max = my_mode_layer_max + 1

    if my_mode_layer_max >= 0:
        _t_a_list  = _t_a_list_raw[:_t_real_max]
    else:
        _t_a_list = _t_a_list_raw
    if len(_t_a_list_raw) > _t_real_max:
        if my_mode_dot_graph_draw_node_basename_level > 0:
            _t_b = '..' + _t_b
    _t_a_list = [ my_identity + '__' + my_ver ] + _t_a_list
    return _t_a_list, _t_b

def get_g_node_body_label_by_attr(layer_last='', op='', stream_id=None, id=None):
    _t_op = '<' + op + '>'
    if my_mode_dot_graph_draw_node_basename_level == 0:
        #_t = op
        _t = _t_op
    else:
        if my_mode_dot_graph_draw_node_basename_level == 1:
            _t_sp = ' '
        else:
            _t_sp = '\n'
        if layer_last:
            _t = _t_sp.join([layer_last, _t_op])
        else:
            _t = _t_op

    # TODO display stream_id alternatively
    if my_mode_node_info_id:
        if id and id != MY_EMPTY_STR:
            _t = id + '_' + _t
        if stream_id:
            _t = (stream_id if stream_id != MY_EMPTY_STR else '-') + '_' + _t
    return _t

def get_g_node_body_tooltip_by_attr(uid, node):
    _t = ''.join([uid + ':' + node['tvm_blockdim'] + '\n' + node['name_strip']])
    _t = _t + '\n<' + node['info_in'] + '\n<(' + node['info_in_orig'] + ')\n>' + node['info_out'] + '\n>(' + node['info_out_orig'] + ')'
    for k in node['prev_node_data_out_dict'].keys():
        _o = node['prev_node_data_out_dict'][k]
        _t = _t + '\n' + k + '<' + _o[1] + ' ' + node_name_strip(_o[0])
    for k in node['next_node_data_in_list_dict'].keys():
        _t_list = node['next_node_data_in_list_dict'][k]
        for _o in _t_list:
            _t = _t + '\n' + k + '>' + _o[1] + ' ' + node_name_strip(_o[0])

    _t = _t[:MY_GRAPH_TOOLTIP_LEN_LIMIT] + '\n.'
    return _t

def get_g_node_body_color_by_attr(node, is_faked=False):
    _t = None
    if node['tvm_blockdim'] and node['tvm_blockdim'] != MY_EMPTY_STR and ((int(node['tvm_blockdim']) % 32) == 0):
        _t = 'dimgray'
    return _t

def get_g_node_body_fill_color_by_attr(node, is_faked=False):
    _gid = my_get_node_highlight_gid(node)
    _t = [None, 'lightcyan', 'lavenderblush', 'lightgray'][_gid]

    return _t

def get_g_node_body_by_attr(uid, label, tooltip, color=None, fillcolor=None, size=None):
    _opt_fontcolor = ',fontcolor=' + color if color else ''
    _opt_fillcolor = ',fillcolor=' + fillcolor if fillcolor else ''
    _opt_fontsize = ',fontsize=' + size if size else ''
    _opt_node_attr = ',xlabel=' + '" "'
    _opt_node_attr = ''
    _t = ''.join(['"', uid, '" [label="', label, '" tooltip="', tooltip, '"', _opt_fontcolor, _opt_fillcolor, _opt_fontsize, _opt_node_attr, ']; '])
    return _t

def get_g_edge_color_by_attr(dtype=''):
    _t = None
    if 'FLOAT16' in dtype or 'HALF' in dtype:
        _t = None   # keep default
    elif 'FLOAT' in dtype:
        _t = 'steelblue'
    else:
        _t = 'deepskyblue'
    return _t

def node_layout_strip(layout):
    _t = layout.replace('FRACTAL_', '') if layout else ''
    return _t

def get_g_edge_label_by_attr(node_in):
    _t = ''
    if my_mode_dot_graph_draw_shape_level == 1:
        _t = node_layout_strip(node_in['layout'])
    elif my_mode_dot_graph_draw_shape_level == 2:
        _t = ' '.join([node_layout_strip(node_in['layout']), node_in['shape']])
    elif my_mode_dot_graph_draw_shape_level == 3:
        _t = ' '.join([node_layout_strip(node_in['layout']), node_in['shape']])
        _t = _t + '\n' + ' '.join([node_layout_strip(node_in['layout_orig']), node_in['shape_orig']])
    elif my_mode_dot_graph_draw_shape_level == 4:
        _t = ' '.join([node_layout_strip(node_in['layout_orig']), node_in['shape_orig']])
    return _t

def get_g_edge_weight_by_attr(stream_id, task_id):
    return None        # FIXME not implemented, which is confusion

    _t = '0'
    if stream_id and stream_id != MY_EMPTY_STR:
        if task_id and task_id != MY_EMPTY_STR:
            _t_s = int(stream_id)
            _t_t = int(task_id)
            #_t = (my_node_stream_id_max_num - _t_s + 1) * my_node_id_max_num - _t_t
            _t = my_node_stream_id_max_num * my_node_id_max_num + _t_t
            _t = str(_t)

    return _t

def get_g_edge_by_attr(uid_from, uid_to, off_from, off_to, label=None, color=None, weight=None, constraint=None):
    # XXX edge attr need careful attention
    _opt_color = ',color=' + color if color is not None else ''
    _opt_label = ',label="' + label + '"' if label is not None else ''
    _opt_weight = ',weight=' + str(weight) if weight is not None else ''
    _opt_constraint = ',constraint=' + str(constraint) if constraint else ''    # TODO
    _opt_edge_attr = ',xlabel=' + '" "'
    _opt_edge_attr = ''
    _t = ''.join(['"', uid_from, '" -> "', uid_to, '" [headlabel=', off_to, ',taillabel=', off_from, _opt_label, _opt_color, _opt_edge_attr, _opt_weight, _opt_constraint, '];'])
    return _t


# TODO: add control edges, and add option to enable/disable this
MY_DEBUG_PRINT('build graph node and edge lines')
my_g_nodes = []
my_g_edges = []
for _node in my_nodes_selected.values():
    # build graph node
    _node_uid = _node['uid']
    _node_layer_list, _node_layer_last = get_layer_list_by_node_name_strip(_node['name_strip'])
    _g_node_begin = ''
    _g_node_end = ''
    for _node_layer in _node_layer_list:
        _g_node_begin = ''.join([_g_node_begin, 'subgraph cluster_' + _node_layer + ' { label="' + _node_layer + '"; '])
        _g_node_end = ''.join([_g_node_end, '} '])
    _g_node_label = get_g_node_body_label_by_attr(_node_layer_last, _node['op_reformat'], _node['stream_id'], _node['id'])
    _g_node_tooltip = get_g_node_body_tooltip_by_attr(_node_uid, _node)
    _g_node_color = get_g_node_body_color_by_attr(_node)
    _g_node_fill_color = get_g_node_body_fill_color_by_attr(_node)
    _g_node_body = get_g_node_body_by_attr(_node_uid, _g_node_label, _g_node_tooltip, color=_g_node_color, fillcolor=_g_node_fill_color)
    _g_node = ''.join([_g_node_begin, _g_node_body, _g_node_end])
    my_g_nodes.append(_g_node)

    # build graph edge prev->curr
    for k in _node['prev_node_data_out_dict'].keys():
        _is_faked_prev_node = False
        _t = _node['prev_node_data_out_dict'][k]
        if _t[0] == MY_EMPTY_STR:
            continue
        if _t[0] in my_nodes_selected:
            _prev_node = my_nodes_selected[_t[0]]
            _from_uid = _prev_node['uid']
        elif not is_node_blocked(my_nodes[_t[0]]):
            _is_faked_prev_node = True
            assert(_t[0] in my_nodes)
            _prev_node = my_nodes[_t[0]]
            _prev_node_fake_uid = _prev_node['uid']
            # XXX hack real layers to make graph more simple
            _prev_node_layer_last = ''
            _g_prev_node_label = get_g_node_body_label_by_attr(_prev_node_layer_last, _prev_node['op_reformat'], _prev_node['stream_id'], _prev_node['id'])
            _g_prev_node_tooltip = get_g_node_body_tooltip_by_attr(_prev_node_fake_uid, _prev_node)
            _g_prev_node_color = get_g_node_body_color_by_attr(_prev_node, True)
            _g_prev_node_fill_color = get_g_node_body_fill_color_by_attr(_prev_node, True)
            _g_prev_node_body = get_g_node_body_by_attr(_prev_node_fake_uid, _g_prev_node_label, _g_prev_node_tooltip, color=_g_prev_node_color, size='7', fillcolor=_g_prev_node_fill_color)
            _g_prev_node_faked = ''.join([_g_node_begin, _g_prev_node_body, _g_node_end])
            my_g_nodes.append(_g_prev_node_faked)

            # TODO
            _from_uid = _prev_node_fake_uid
        else:
            continue
        _to_uid = _node_uid
        _from_off = _t[1]
        _to_off = k

        _tmp = _node['in_data_dict'][_to_off] # dtype, layout, shape
        _e_color = get_g_edge_color_by_attr(dtype=_tmp['dtype'])
        _e_label = get_g_edge_label_by_attr(_tmp)
        _e_weight = get_g_edge_weight_by_attr(_node['stream_id'], _node['id'])
        _e = get_g_edge_by_attr(_from_uid, _to_uid, _from_off, _to_off, color=_e_color, label=_e_label, weight=_e_weight)
        my_g_edges.append(_e)

    #build graph edge curr->next
    for k in _node['next_node_data_in_list_dict'].keys():
        _t_list = _node['next_node_data_in_list_dict'][k]
        for _t in _t_list:
            _is_faked_next_node = False
            if _t[0] == MY_EMPTY_STR:
                continue
            if _t[0] in my_nodes_selected:
                _next_node = my_nodes_selected[_t[0]]
                _to_uid = _next_node['uid']
            elif not is_node_blocked(my_nodes[_t[0]]):
                #continue    # FIXME do not display non selected next node

                _is_faked_next_node = True
                assert(_t[0] in my_nodes)
                _next_node = my_nodes[_t[0]]
                _next_node_fake_uid = _next_node['uid']
                # XXX hack real layers to make graph more simple
                _next_node_layer_last = ''
                _g_next_node_label = get_g_node_body_label_by_attr(_next_node_layer_last, _next_node['op_reformat'], _next_node['stream_id'], _next_node['id'])
                _g_next_node_tooltip = get_g_node_body_tooltip_by_attr(_next_node_fake_uid, _next_node)
                _g_next_node_color = get_g_node_body_color_by_attr(_next_node, True)
                _g_next_node_fill_color = get_g_node_body_fill_color_by_attr(_next_node, True)
                _g_next_node_body = get_g_node_body_by_attr(_next_node_fake_uid, _g_next_node_label, _g_next_node_tooltip, color=_g_next_node_color, size='7', fillcolor=_g_next_node_fill_color)
                _g_next_node_faked = ''.join([_g_node_begin, _g_next_node_body, _g_node_end])
                my_g_nodes.append(_g_next_node_faked)

                # TODO
                _to_uid = _next_node_fake_uid
            else:
                continue
            _to_off = _t[1]
            _from_uid = _node_uid
            _from_off = k

            _tmp = _next_node['in_data_dict'][_to_off] # dtype, layout, shape
            _e_color = get_g_edge_color_by_attr(dtype=_tmp['dtype'])
            _e_label = get_g_edge_label_by_attr(_tmp)
            _e_weight = get_g_edge_weight_by_attr(_next_node['stream_id'], _next_node['id'])
            _e = get_g_edge_by_attr(_from_uid, _to_uid, _from_off, _to_off, color=_e_color, label=_e_label, weight=_e_weight)
            my_g_edges.append(_e)
            
    # TODO build graph edge prev->curr ctrl

    # TODO build graph edge curr->next ctrl


MY_DEBUG_PRINT('build graph')
my_g_edges = sorted(list(set(my_g_edges)))  # uniq the edge

my_g = []
my_g.append('digraph ' + my_identity + '_' + my_ver + ' {')
#my_g.append('label="' + my_identity + '_' + my_ver + '";')    # TODO
my_g.append('node [color=dimgray,penwidth=0.1,shape=plain,style="rounded,filled",fillcolor=whitesmoke,fontcolor=navy,fontsize=10,fontname="Arial Narrow",height=0,width=0,margin=0.0];')
my_g.append('edge [color=lightblue,penwidth=0.5,arrowhead=vee,arrowsize=0.3,minlen=1,labelfontcolor=gray,labelfontname="Arial Narrow",labelfontsize=6,decorate=true,fontcolor=gray,fontname="Arial Narrow",fontsize=6];')
my_g.append('graph [' + my_mode_dot_graph_attr + ',color=gray,penwidth=0.1,pencolor=limegreen,fontcolor=gray,fontsize=6,fontname="Arial Narrow",labeljust=l,margin=0.0,nodesep=0.05,ranksep=0.2,splines=true,newrank=true,mclimit=1.0];')
my_g.append('')
for o in my_g_nodes:
    my_g.append(o)
for o in my_g_edges:
    my_g.append(o)
my_g.append('')
my_g.append('}')


MY_DEBUG_PRINT('write to dot file:', my_fn_out_dot)
with open(my_fn_out_dot, 'wb') as f:
    for o in my_g:
        f.write(''.join([o, '\n']).encode('ascii'))

MY_DEBUG_PRINT('# tips: generate figure by command below')
for t in ['svg', 'pdf', 'png']:
    MY_DEBUG_PRINT('dot -T' + t + ' -o ' + my_fn_out_dot + '.' + t + ' ' + my_fn_out_dot)

MY_DEBUG_PRINT('# end')
