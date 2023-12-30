CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port 29716 train_final_dist.py --train_data  /your/train_data/path \
--valid_data /your/valid_data/path  --select_data data_name \
-batch_ratio 1  --Transformer ACDS-STR-Base  --imgH 32 --imgW 128 \
--manualSeed=226 --workers=4 --isrand_aug --scheduler --batch_size 80 --rgb --is_mask_supervision \
--saved_model  /your/pretrain_model/path \
--saved_path  --exp_name sign --valInterval 5000 --num_iter 2000000 --lr 1
