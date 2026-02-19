#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp26_exp26_ema_weights_eval/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp26_exp26_ema_weights_eval \
  --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --model d12 --batch_size 4 --grad_accumulation_steps 1 --sequence_length 64 --num_iterations 4768 --learning_rate 0.0001 --warmup_iters 256 --warmdown_iters 2048 --weight_decay 0.0 --val_loss_every 128 --val_batch_size 16 --ema_decay 0.993
