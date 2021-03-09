#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko \
    --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.ko \
    --mono-en=./en_es_data/train_mono.en --mono-ko=./en_es_data/train_mono.ko --vocab=vocab.json \
    --map-en=./net_Module/mapping_en.pth --map-ko=./net_Module/mapping_ko.pth \
    --slang=en --tlang=ko --batch_ratio=2 --loss_ratio=1 --self_learning=0 --mapping=0 --map_learning=0 \
    --back_trans=1 --cuda
elif [ "$1" = "test" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.en ./en_es_data/test.ko outputs/test_en2ko.txt \
    --slang=en --tlang=ko --cuda
elif [ "$1" = "backtr" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py backtr model.bin ./en_es_data/test.en ./en_es_data/test.ko outputs/bcktr_en2ko.txt \
    --slang=en --tlang=ko --mapping=0 --batch-size=64 
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.ko --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.en ./en_es_data/test.ko outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko vocab.json
else
	echo "Invalid Option Selected"
fi