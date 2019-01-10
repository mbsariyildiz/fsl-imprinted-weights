n_epochs=100
batch_size=64
lr=0.0001
lr_decay=0.94
lr_decay_steps=4
device='cuda'

n_shotss=(1 2 5 10 20)
seeds=(11 22 33 44 55)
n_data_augments=5

data_dir=../data/CUB_200_2011
model_ckpt_path=$EXPERIMENTS_HOME/fsl-imprinted-weights/base_classifier/model.ckpt

################################################################################

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/imprinting_ft
for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		exp_dir=$exp_root/$(printf "n-shots_%02d/%02d" $n_shots $exp_iter)
		seed=${seeds[$((exp_iter-1))]}
		
		echo ''
		echo ''
		echo '********************************************************************************'
		echo 'imprinting with samples of novel classes and then fine-tuning'
		echo seed: $seed
		echo data-dir: $data_dir
		echo exp-dir: $exp_dir
		echo model-ckpt-path: $model_ckpt_path

		python ../src/imprint_ft.py \
			--device=$device \
			--exp_dir=$exp_dir \
			--data_dir=$data_dir \
			--model_ckpt_path=$model_ckpt_path \
			--n_epochs=$n_epochs \
			--batch_size=$batch_size \
			--lr=$lr \
			--lr_decay=$lr_decay \
			--lr_decay_steps=$lr_decay_steps \
			--n_shots=$n_shots \
			--seed=$seed \
			--n_data_augments=$n_data_augments

	done

	exp_root_dir=$exp_root/$(printf "n-shots_%02d" $n_shots)
	keys='novel_recall,base_recall,avg_recall,per_class_recall'
	python ../src/combine_results.py \
		--exp_root_dir=$exp_root_dir \
		--keys=$keys \
		--verbose

done

################################################################################

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/imprinting_random_ft
for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		exp_dir=$exp_root/$(printf "n-shots_%02d/%02d" $n_shots $exp_iter)
		seed=${seeds[$((exp_iter-1))]}

		echo ''
		echo ''
		echo '********************************************************************************'
		echo 'imprinting with random weights and then fine-tunining'
		echo seed: $seed
		echo data-dir: $data_dir
		echo exp-dir: $exp_dir
		echo model-ckpt-path: $model_ckpt_path

		python ../src/imprint_ft.py \
			--device=$device \
			--exp_dir=$exp_dir \
			--data_dir=$data_dir \
			--model_ckpt_path=$model_ckpt_path \
			--n_epochs=$n_epochs \
			--batch_size=$batch_size \
			--lr=$lr \
			--lr_decay=$lr_decay \
			--lr_decay_steps=$lr_decay_steps \
			--n_shots=$n_shots \
			--seed=$seed \
			--n_data_augments=$n_data_augments \
			--random

	done

	exp_root_dir=$exp_root/$(printf "n-shots_%02d" $n_shots)
	keys='novel_recall,base_recall,avg_recall,per_class_recall'
	python ../src/combine_results.py \
		--exp_root_dir=$exp_root_dir \
		--keys=$keys \
		--verbose

done
