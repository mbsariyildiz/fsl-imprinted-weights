n_data_augments=5
batch_size=64
d_emb=64
device='cuda'
n_shotss=(1 2 5 10 20)
seeds=(11 22 33 44 55)

exp_dir=$EXPERIMENTS_HOME/fsl-imprinted-weights/base_classifier
data_dir=../data/CUB_200_2011

for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		seed=${seeds[$((exp_iter-1))]}

		fts_save_path=$exp_dir/features/$(printf "features_aug-%02d_n-%02d_%02d" $n_data_augments $n_shots $exp_iter)
		ws_save_path=$exp_dir/weights/$(printf "weights_n-%02d_%02d" $n_shots $exp_iter)

		mkdir -p $exp_dir/features
		mkdir -p $exp_dir/weights

		echo ""
		echo "**************************************************"
		echo seed: $seed
		echo exp-dir: $exp_dir
		echo data-dir: $data_dir
		echo fts-save-path: $fts_save_path
		echo ws-save-path: $ws_save_path

		python ../src/extract_features.py \
			--data_dir=$data_dir \
			--exp_dir=$exp_dir \
			--fts_save_path=$fts_save_path \
			--ws_save_path=$ws_save_path \
			--device=$device \
			--n_workers=8 \
			--n_shots=$n_shots \
			--seed=$seed \
			--n_data_augments=$n_data_augments \
			--batch_size=$batch_size \
			--d_emb=$d_emb \
			--cosine_sim

	done
done