#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp9_exp9_adaptive_attention_scale/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp9_exp9_adaptive_attention_scale \
  --batch_size=4 --sequence_length=64 --num_iterations=4768 --learning_rate=0.0001 --warmup_iters=0 --warmdown_iters=1430 --weight_decay=0 --val_loss_every=128 --val_batch_size=16 --model=d12
