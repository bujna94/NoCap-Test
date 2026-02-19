#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp5_progressive_sequence_length/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp5_progressive_sequence_length \
  --input_bin=data/fineweb10B/fineweb_train_*.bin --input_val_bin=data/fineweb10B/fineweb_val_*.bin --model=d12 --batch_size=8 --sequence_length=64 --num_iterations=4768 --learning_rate=0.0006 --warmup_iters=400 --warmdown_iters=1268 --weight_decay=0.0 --val_loss_every=128 --val_batch_size=16
