# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from queue import SimpleQueue

import mxnet as mx
from mxnet import cuda_utils as cu
from mlperf_logger import mllog_event as log_event


class ScaleoutBridge(object):
    FWD_TIME = 'fwd_time'
    BWD_TIME = 'bwd_time'
    OPT_TIME = 'opt_time'
    LOAD_TIME = 'load_time'
    EVAL_TIME = 'eval_time'
    EPOCH_TIME = 'epoch_time'

    def __init__(self, qmax):
        print("Scaleout performance bridge is running ...")
        self.qmax = qmax
        self.fwdq = SimpleQueue()
        self.bwdq = SimpleQueue()
        self.optq = SimpleQueue()
        self.loadq = SimpleQueue()
        self.evalq = SimpleQueue()

    def push_nvtx(self, tag):
        cu.nvtx_range_push(tag)

    def pop_nvtx(self):
        cu.nvtx_range_pop()

    def print_tag(self, tag, dur, deviceid):
        log_event(key=tag, value={'r':deviceid, 't':dur}, uniq=False)

    def add_tag(self, tag, dur, deviceid):
        if self.fwdq.qsize() > self.qmax:
            self.empty_qs()
            return 0
        if tag == self.FWD_TIME:
            self.fwdq.put((dur, deviceid))
        elif tag == self.BWD_TIME:
            self.bwdq.put((dur, deviceid))
        elif tag == self.OPT_TIME:
            self.optq.put((dur, deviceid))
        elif tag == self.LOAD_TIME:
            self.loadq.put((dur, deviceid))
        elif tag == self.EVAL_TIME:
            self.evalq.put((dur, deviceid))
        else:
            assert("Tag not supported" and 0)
        return 1

    def empty_qs(self):
        self.empty_q(self.fwdq, self.FWD_TIME)
        self.empty_q(self.bwdq, self.BWD_TIME)
        self.empty_q(self.optq, self.OPT_TIME)
        self.empty_q(self.loadq, self.LOAD_TIME)
        self.empty_q(self.evalq, self.EVAL_TIME)

    def empty_q(self, q, tag):
        while not q.empty():
            atuple = q.get()
            log_event(key=tag, value={'r': atuple[1], 't': atuple[0]}, uniq=False)

def init_bridge():
    time_tags = int(os.getenv('TIME_TAGS', 0))
    nvtx_flag = int(os.getenv('NVTX_FLAG', 0))
    sbridge = None
    time_start = 0
    if time_tags or nvtx_flag:
        sbridge = ScaleoutBridge(300)
        mx.nd.waitall()
        time_start = time.time()

    return time_tags, nvtx_flag, sbridge, time_start

