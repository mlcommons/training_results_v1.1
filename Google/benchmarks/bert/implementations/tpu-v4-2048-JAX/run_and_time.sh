#!/bin/bash
python3 lingvo/jax/mlperf/main.py -- \
    --model=lingvo.lm.bert.BertSpmdL66H12kBiggerBatch8x8x16 \
    --multi_host_checkpointing=True \
    --job_log_dir=<log_dir> \
    --restore_checkpoint_dir=<init_ckpt_dir> \
    --restore_checkpoint_step=<init_ckpt_step> \
    --tasks_per_host=4 \
    --max_train_steps=29000 \
    --target_accuracy=0.80 \
    --globally_use_hardware_rng \
    --eval_on_test
