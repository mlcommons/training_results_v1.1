import re
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('path',
                    type=str,
                    default="/mlperf/boh/workspace/beta/resnet50_0427/logs_dali_single2/",
                    help='base file path')
FLAGS = parser.parse_args()
print("Path:", FLAGS.path)

_RUN_START_REGEX = r':::MLLOG (.*"run_start",.*)'
_RUN_STOP_REGEX = r':::MLLOG (.*"run_stop",.*)'


for root, dirs, files in os.walk(FLAGS.path):
    avg_minute = 0
    num = 0
    for f in files:
        path = os.path.join(root, f)
        with open(path, 'r',encoding='gbk') as file:
            result = file.read()
            run_start = re.search(_RUN_START_REGEX, result)

            try:
                run_start = json.loads(run_start.group(1))['time_ms']
            except:
                print("no run start")
                continue
            run_stop = re.search(_RUN_STOP_REGEX, result)
            try:
                run_stop = json.loads(run_stop.group(1))['time_ms']
            except:
                print("no run stop")
                continue
            seconds = float(run_stop) - float(run_start)
            minutes = seconds / 60 / 1000 # convert ms to minutes
            print(path, " time:", minutes)
            avg_minute += minutes
            #print(minutes)
            num += 1
    print("avg_time:", avg_minute/num)
