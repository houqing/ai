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

def my_pt(*kwargs):
    print('TFDBG_DMP]', *kwargs)

if len(sys.argv) == 2:
    start_cmd = sys.argv[1]
else:
    my_pt(' '.join(['Usage:', sys.argv[0], '<tfdbg_enabled_model_start_cmd>']))
    exit()

my_pt('start the command and enter tf debug cli:', start_cmd)
f_log = open(sys.argv[0]+'-'+my_ts+'.log', 'wb')
# FIXME: set timeout=None to disable the timeout check
c = pexpect.spawn(start_cmd, timeout=None, logfile=f_log)
c.expect_exact('tfdbg> ')


def run_and_dump(iter_num, dump_dir="dump-data"):
    _tensor_list_file = dump_dir + '--tensor.list'

    cmd = 'run -t ' + str(iter_num)
    my_pt('run number of iters:', cmd)
    c.sendline(cmd)
    c.expect_exact('dumped tensor(s):')
    c.expect_exact('tfdbg> ')

    cmd = 'lt > ' + _tensor_list_file
    my_pt('get tensor list into file:', cmd)
    c.sendline(cmd)
    c.expect_exact('INFO: Wrote output to ')
    c.expect_exact('tfdbg> ')

    my_pt('create dump dir, remove firstly if exist:', dump_dir)
    shutil.rmtree(dump_dir, ignore_errors=True)
    time.sleep(0.1)
    os.makedirs(dump_dir, exist_ok=True)
    time.sleep(1)

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
                c.sendline()
                c.expect_exact('tfdbg> ', timeout=3600)
                time.sleep(0.005)
                c.sendline(cmd)
                #c.expect_exact('Saved value to:', timeout=3600)
                _id = c.expect_exact(['tfdbg> ', pexpect.TIMEOUT], timeout=3600)
                if _id == 0:
                    time.sleep(0.005)
                else:
                    print('!', end='', flush=True)
                c.sendline()
                c.expect_exact('tfdbg> ', timeout=3600)
                time.sleep(0.005)
            else:
                print('-', end='', flush=True)
            _cnt += 1
        print('', flush=True)

    my_pt('dump tensors end')

_iter_executed = 0
_iter_list = [1, 2, 3125, 3126]

for _iter_offset in _iter_list:
    _iter_todo = _iter_offset - _iter_executed
    run_and_dump(_iter_todo, 'dump-data-' + str(_iter_offset))
    _iter_executed += _iter_todo

