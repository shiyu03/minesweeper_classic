#!/usr/bin/env bash
eval "$(/home/nsp/anaconda3/bin/conda shell.bash hook)"
conda activate zzj_ms
tensorboard --logdir runs --host 0.0.0.0
