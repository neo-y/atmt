#!/bin/bash
# -*- coding: utf-8 -*-

set -e

n_symbols=$1
threshold=$2
pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed_$1_$2/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed_$1_$2/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed_$1_$2/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed_$1_$2/tm.$src --corpus $data/preprocessed_$1_$2/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed_$1_$2/tm.$tgt --corpus $data/preprocessed_$1_$2/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed_$1_$2/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$src > $data/preprocessed_$1_$2/train_truecased.$src
cat $data/preprocessed_$1_$2/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$tgt > $data/preprocessed_$1_$2/train_truecased.$tgt


# apply bpe model
# train bpe
subword-nmt learn-joint-bpe-and-vocab --input $data/preprocessed_$1_$2/train_truecased.$src $data/preprocessed_$1_$2/train_truecased.$tgt -s $1 -o $data/preprocessed_$1_$2/bpe_codes --write-vocabulary $data/preprocessed_$1_$2/vocab.$src $data/preprocessed_$1_$2/vocab.$tgt

# apply bpe (encoding)
if $2 == 0
then
subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$src < $data/preprocessed_$1_$2/train_truecased.$src > $data/preprocessed_$1_$2/train.$src
subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$tgt < $data/preprocessed_$1_$2/train_truecased.$tgt > $data/preprocessed_$1_$2/train.$tg
else
subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$src --vocabulary-threshold $2 < $data/preprocessed_$1_$2/train_truecased.$src > $data/preprocessed_$1_$2/train.$src
subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$tgt --vocabulary-threshold $2 < $data/preprocessed_$1_$2/train_truecased.$tgt > $data/preprocessed_$1_$2/train.$tgt
fi

# prepare remaining splits with learned models
if $2 == 0
then
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$src | subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$src > $data/preprocessed_$1_$2/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$tgt | subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$tgt > $data/preprocessed_$1_$2/$split.$tgt
done
else
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$src | subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$src --vocabulary-threshold $2 > $data/preprocessed_$1_$2/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_$1_$2/tm.$tgt | subword-nmt apply-bpe -c $data/preprocessed_$1_$2/bpe_codes --vocabulary $data/preprocessed_$1_$2/vocab.$tgt --vocabulary-threshold $2 > $data/preprocessed_$1_$2/$split.$tgt
done
fi

# remove tmp files
rm $data/preprocessed_$1_$2/train.$src.p
rm $data/preprocessed_$1_$2/train.$tgt.p
rm $data/preprocessed_$1_$2/train_truecased.$src
rm $data/preprocessed_$1_$2/train_truecased.$tgt


# preprocess all files for model training
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared_$1_$2/ --train-prefix $data/preprocessed_$1_$2/train --valid-prefix $data/preprocessed_$1_$2/valid --test-prefix $data/preprocessed_$1_$2/test --tiny-train-prefix $data/preprocessed_$1_$2/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 --vocab-src $data/preprocessed_$1_$2/vocab.$src --vocab-trg $data/preprocessed_$1_$2/vocab.$tgt

# copy vocab files to
cp $data/preprocessed_$1_$2/vocab.$src $data/prepared_$1_$2/dict.$src
cp $data/preprocessed_$1_$2/vocab.$tgt $data/prepared_$1_$2/dict.$tgt

echo "done!"
