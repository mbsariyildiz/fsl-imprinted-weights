n_data_augments=10
batch_size=64
d_emb=64
device='cuda'

exp_dir=$EXPERIMENTS_HOME/fsl-imprinted-weights/base_pretrained_dot
data_dir=../data/CUB_200_2011
echo exp-dir: $exp_dir
echo data-dir: $data_dir

python ../src/extract_features.py \
	--data_dir=$data_dir \
	--exp_dir=$exp_dir \
	--device=$device \
	--n_workers=8 \
	--n_data_augments=$n_data_augments \
	--batch_size=$batch_size \
	--d_emb=$d_emb
	# --cosine_sim

