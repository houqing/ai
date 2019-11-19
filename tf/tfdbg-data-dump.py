#/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import pexpect
import sys
import os
import shutil
import time

my_ts = time.strftime('%Y%m%d-%H%M%S')
my_tfdbg_prompt = 'tfdbg> '

def my_pt(*kwargs):
    print('TFDBG_DMP]', *kwargs)

def tfdbg_cleanup_residual_prompt(timeout=0.010):
    while True:
        _id = c.expect_exact([my_tfdbg_prompt, pexpect.TIMEOUT], timeout=timeout)
        if _id == 1:
            break

def tfdbg_sendline_expect_safe(client, cmd, exp=None, timeout=-1, is_ignore_timeout=False):
    tfdbg_cleanup_residual_prompt()
    if cmd != None:
        client.sendline(cmd)
    try:
        if exp:
            client.expect_exact(exp, timeout)
        client.expect_exact(my_tfdbg_prompt, timeout)
    except pexpect.TIMEOUT:
        if is_ignore_timeout:
            return 1
        else:
            client.expect_exact(my_tfdbg_prompt, 0.001)

    return 0


if len(sys.argv) == 2:
    start_cmd = sys.argv[1]
else:
    my_pt(' '.join(['Usage:', sys.argv[0], '<tfdbg_enabled_model_start_cmd>']))
    exit()

my_pt('start the command and enter tf debug cli:', start_cmd)
f_log = open(sys.argv[0]+'-'+my_ts+'.log', 'wb')
# FIXME: set timeout=None to disable the timeout check
c = pexpect.spawn(start_cmd, timeout=None, logfile=f_log)
c.setwinsize(25, 4096)  # row, col
tfdbg_sendline_expect_safe(c, None)

def run_and_dump(iter_num, dump_dir="dump-data"):
    _tensor_list_file = dump_dir + '--tensor.list'

    cmd = 'run -t ' + str(iter_num)
    my_pt('run number of iters:', cmd)
    tfdbg_sendline_expect_safe(c, cmd, 'dumped tensor(s):')

    cmd = 'lt > ' + _tensor_list_file
    my_pt('get tensor list into file:', cmd)
    tfdbg_sendline_expect_safe(c, cmd, 'INFO: Wrote output to ')

    my_pt('create dump dir, remove firstly if exist:', dump_dir)
    shutil.rmtree(dump_dir, ignore_errors=True)
    os.makedirs(dump_dir, exist_ok=True)
    time.sleep(0.010)

    my_pt('dump tensors begin:', _tensor_list_file)
    with open(_tensor_list_file) as f_list:
        _cnt = 0
        for line in f_list:
            if (_cnt % 100) == 0:
                print('', flush=True)
                print('%3d: ' % (_cnt // 100), end='', flush=True)
            _tensor_name_raw = line.rstrip().split(' ')[-1]
            if any([_s in _tensor_name_raw for _s in [':0',':1',':2',':3',':4',':5',':6',':7',':8',':9']]):
                print(_cnt % 10, end='', flush=True)
                _tensor_name_fixed = _tensor_name_raw.replace('/', '-').replace(':', '-')
                cmd = ' '.join(['pt -s', _tensor_name_raw, '-w', dump_dir+'/'+_tensor_name_fixed+'.npy'])
                _id = tfdbg_sendline_expect_safe(c, cmd, timeout=3600, is_ignore_timeout=True)
                if _id != 0:
                    print('!', end='', flush=True)
            else:
                print('-', end='', flush=True)
            _cnt += 1
        print('', flush=True)

    my_pt('dump tensors end')

_iter_executed = 0
_iter_list = [1, 2, 3125, 3126, 19200, 19201, 38300, 38301]
#_iter_list = [1, 2, 19200, 19201, 38300, 38301]

for _iter_offset in _iter_list:
    _iter_todo = _iter_offset - _iter_executed
    run_and_dump(_iter_todo, 'dump-data-' + str(_iter_offset))
    _iter_executed += _iter_todo

c.interact()

