#!/bin/bash 
# srun --job-name=test  --gres=gpu:4  -A J00120230004 --reservation=root_114 -w pgpu18 -p p-A100 -c 48  --pty bash
srun --job-name=test  --gres=gpu:1  -A J00120230004 -w pgpu27 -p p-A800 -c 12  --pty bash