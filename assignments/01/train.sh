#! /bin/bash


python ../../train.py \
--cuda TRUE \
--data ../../data/en-sv/infopankki/prepared \
--source-lang sv \
--target-lang en \
--log-file ./train_output_cho.txt \
--save-dir ../../assignments/01/baseline/checkpoints

