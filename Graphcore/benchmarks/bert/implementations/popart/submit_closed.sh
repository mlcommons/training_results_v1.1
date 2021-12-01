#!/bin/bash

make

POD_SIZE=$1

export POPLAR_ENGINE_OPTIONS='{"target.syncReplicasIndependently": "true", "target.hostSyncTimeout": "3000"}' 

python3 ./create_submission.py --pod $POD_SIZE --submission-division closed
