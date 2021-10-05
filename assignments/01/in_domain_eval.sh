#! /bin/bash

python ../../translate.py \
--data ../../data/en-sv/infopankki/prepared \
--dicts ../../data/en-sv/infopankki/prepared \
--checkpoint-path baseline/checkpoints/checkpoint_last.pt \
--output baseline/infopankki_translations.txt
