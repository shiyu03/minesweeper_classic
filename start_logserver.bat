@echo off
call conda activate msoffice
tensorboard --logdir runs --host 0.0.0.0
