#/bin/bash
python train/test.py --gpu 0 --num_point 3500 --model frustum_pointnets_v2 --model_path '/root/frustum-pointnets_RSC_2D_reorg/train/log_v2/08-07-2020-13:57:37/ckpt/model_150.ckpt' --batch_size 1 --output train/detection_results_v2 --idx_path kitti/image_sets/val.txt
