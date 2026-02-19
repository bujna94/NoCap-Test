#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp2_dynamic_grad_accumulation/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp2_dynamic_grad_accumulation \
  --model d12 --batch_size 16 --sequence_length 1024 --num_iterations 4768 --learning_rate 0.0006 --warmup_iters 0 --warmdown_iters 1430 --weight_decay 0.0 --val_loss_every 128 --val_batch_size 16 --grad_accumulation_steps 1
