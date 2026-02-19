#!/bin/bash
cd /workspace/NoCap-Test
torchrun --standalone --nproc_per_node=1 experiments/exp1_adaptive_layer_lr/train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir experiments/exp1_adaptive_layer_lr \
  --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --model d12 --batch_size 16 --sequence_length 1024 --num_iterations 4768 --learning_rate 0.0006 --warmup_iters 50 --warmdown_iters 953 --weight_decay 0.0 --val_loss_every 128 --val_batch_size 16
