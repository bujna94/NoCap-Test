#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp24_exp24_cosine_schedule_higher_lr/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp24_exp24_cosine_schedule_higher_lr \
  --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --model d12 --batch_size 4 --sequence_length 64 --num_iterations 4768 --learning_rate 1.2e-4 --warmup_iters 384 --warmdown_iters 1024 --weight_decay 0.0 --val_loss_every 128 --val_batch_size 16 --use_cosine_schedule --min_lr_ratio 0.05
