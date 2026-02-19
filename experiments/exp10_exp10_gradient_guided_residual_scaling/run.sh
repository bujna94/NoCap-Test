#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp10_exp10_gradient_guided_residual_scaling/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp10_exp10_gradient_guided_residual_scaling \
  --model d12 --batch_size 4 --sequence_length 64 --num_iterations 4768 --learning_rate 1e-4 --warmup_iters 0 --warmdown_iters 715 --weight_decay 0 --val_loss_every 128 --val_batch_size 16
