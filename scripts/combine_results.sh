#!/bin/bash

exp_root_dir=../log/imprinting/n-shots_01
keys='test_top1,rec_pc,base_rec,novel_rec'

python ../src/combine_results.py \
	--exp_root_dir=$exp_root_dir \
	--keys=$keys
