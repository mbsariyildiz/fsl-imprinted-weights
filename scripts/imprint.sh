n_data_augments=10
batch_size=128
device='cuda'
n_shotss=(1 2 5 10 20)
seeds=(11 22 33 44 55)

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/imprinting_random
model_ckpt_path=$EXPERIMENTS_HOME/fsl-imprinted-weights/base_pretrained/model.ckpt
data_dir=../data/CUB_200_2011

for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		seed=${seeds[$((exp_iter-1))]}
		echo ''
		echo '********************' seed: $seed '********************'
		
		exp_dir=$exp_root/$(printf "n-shots_%02d/%02d" $n_shots $exp_iter)
		echo data-dir: $data_dir
		echo exp-dir: $exp_dir
		echo model-ckpt-path: $model_ckpt_path

		python ../src/imprint.py \
			--device=$device \
			--exp_dir=$exp_dir \
			--data_dir=$data_dir \
			--model_ckpt_path=$model_ckpt_path \
			--n_data_augments=$n_data_augments \
			--batch_size=$batch_size \
			--n_shots=$n_shots \
			--seed=$seed \
			--random

	done

	exp_root_dir=$exp_root/$(printf "n-shots_%02d" $n_shots)
	keys='test_top1,rec_pc,base_rec,novel_rec'
	python ../src/combine_results.py \
		--exp_root_dir=$exp_root_dir \
		--keys=$keys


done
