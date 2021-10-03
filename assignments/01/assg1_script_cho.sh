#! /bin/bash


python train.py \
--train-on-tiny \
--data ../../data/en-sv/infopankki/prepared \
--source-lang sv \
--target-lang en \
--save-dir ../../assignments/01/baseline/checkpoints > train_output_cho.txt
