import collections
import os
import subprocess
from mlperf_logging.mllog import constants as mlperf_constants

os.environ['PYTHONPATH'] += os.path.join(os.getcwd(),'minigo')
from minigo.ml_perf.logger import log_event, mpiwrapper


def all_reduce(v):
    return mpiwrapper.allreduce(v)

def mlperf_submission_log(benchmark):

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    log_event(
        key=mlperf_constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=mlperf_constants.SUBMISSION_ORG,
        value='Azure')

    log_event(
        key=mlperf_constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=mlperf_constants.SUBMISSION_STATUS,
        value='cloud')

    log_event(
        key=mlperf_constants.SUBMISSION_PLATFORM,
        value='{}xND96amsr_A100_v4'.format(num_nodes))
