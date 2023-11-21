CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 \
train.py \
--batchsize 8 \
--savepath "./model" \
--datapath "../data/DUTS/DUTS-TR" \


