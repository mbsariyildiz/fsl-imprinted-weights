
device='cuda'
n_workers=4
n_epochs=100
batch_size=64
lr=0.001
lr_decay=0.94
lr_decay_steps=4
seeds=(11 22 33 44 55)
n_shotss=(1 2 5 10 20)
data_dir=../data/CUB_200_2011

################################################################################

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/base_classifier
exp_dir=$exp_root

echo ''
echo ''
echo '********************************************************************************'
echo 'training a base classifier (using samples of only the base classes) '
echo exp-dir: $exp_dir
echo data-dir: $data_dir

python ../src/train.py \
	--device=$device \
	--n_workers=$n_workers \
	--exp_dir=$exp_dir \
	--data_dir=$data_dir \
	--n_epochs=$n_epochs \
	--batch_size=$batch_size \
	--lr=$lr \
	--lr_decay=$lr_decay \
	--lr_decay_steps=$lr_decay_steps \
	--batch_size=$batch_size \
	--n_shots=0 \
	--seed=0 \
	--fine_tune \
	--cosine_sim

################################################################################

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/alljoint

for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		exp_name=$(printf "n-shots_%02d/%02d" $n_shots $exp_iter)
		seed=${seeds[$((exp_iter-1))]}

		exp_dir=$exp_root/$exp_name
		data_dir=../data/CUB_200_2011

		echo ''
		echo ''
		echo '********************************************************************************'
		echo 'training an allclassjoint classifier (no cosine similarity)'
		echo exp-dir: $exp_dir
		echo data-dir: $data_dir
		echo seed: $seed

		python ../src/train.py \
			--device=$device \
			--n_workers=$n_workers \
			--exp_dir=$exp_dir \
			--data_dir=$data_dir \
			--n_epochs=$n_epochs \
			--batch_size=$batch_size \
			--lr=$lr \
			--lr_decay=$lr_decay \
			--lr_decay_steps=$lr_decay_steps \
			--batch_size=$batch_size \
			--n_shots=$n_shots \
			--seed=$seed \
			--fine_tune \
			--cosine_sim

	done
done


################################################################################

exp_root=$EXPERIMENTS_HOME/fsl-imprinted-weights/alljoint_cossim

for n_shots in "${n_shotss[@]}"; do
	for exp_iter in `seq 1 1 5`; do

		exp_name=$(printf "n-shots_%02d/%02d" $n_shots $exp_iter)
		seed=${seeds[$((exp_iter-1))]}

		exp_dir=$exp_root/$exp_name
		data_dir=../data/CUB_200_2011

		echo ''
		echo ''
		echo '********************************************************************************'
		echo 'training an allclassjoint-cossim classifier (with cosine similarity)'
		echo exp-dir: $exp_dir
		echo data-dir: $data_dir
		echo seed: $seed

		python ../src/train.py \
			--device=$device \
			--n_workers=$n_workers \
			--exp_dir=$exp_dir \
			--data_dir=$data_dir \
			--n_epochs=$n_epochs \
			--batch_size=$batch_size \
			--lr=$lr \
			--lr_decay=$lr_decay \
			--lr_decay_steps=$lr_decay_steps \
			--batch_size=$batch_size \
			--n_shots=$n_shots \
			--seed=$seed \
			--fine_tune \
			--cosine_sim

	done
done
