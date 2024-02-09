#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_value> [<dataset_value1> ...] [other_parameters]"
    exit 1
fi

model=$1
shift  # Rimuovi il primo parametro dalla lista

if [ $# -eq 0 ]; then
    datasets=("elephant" "tiger" "fox", "musk1", "musk2", "messidor")
else
    datasets=("$@")
fi

for dataset in "${datasets[@]}"; do
    python main.py --model=$model --dataset=$dataset "$@" &
done