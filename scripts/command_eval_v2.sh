#/bin/bash
python train/train_eval.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 1 --batch_size 32 --decay_step 800000 --decay_rate 0.5  --restore_model_path