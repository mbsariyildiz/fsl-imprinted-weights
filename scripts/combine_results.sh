#!/bin/bash

for n_shots in 1 2 5 10 20; do

	exp_root_dir=$EXPERIMENTS_HOME/fsl-imprinted-weights/alljoint/n-shots_$(printf "%02d" $n_shots)
	keys='train_top1,test_top1,best_top1,avg_recall,base_recall,novel_recall'

	echo ""
	echo ""
	python ../src/combine_results.py \
		--exp_root_dir=$exp_root_dir \
		--keys=$keys

done
