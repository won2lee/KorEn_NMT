#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.ko --vocab=vocab.json --slang=en --tlang=ko
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.en ./en_es_data/test.ko outputs/test_en2ko.txt
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko --dev-src=./en_es_data/dev.en --dev-tgt=./en_es_data/dev.ko --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.en ./en_es_data/test.ko outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.en --train-tgt=./en_es_data/train.ko vocab.json
else
	echo "Invalid Option Selected"
fi
