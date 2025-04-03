#!/bin/bash
cuda=$1

reference_path="" # Path to ground truth audios

split_path="samples.txt"   # Path to split txt

result_path=""     # Path to save results

set -x 
CUDA_VISIBLE_DEVICES=${cuda} python eval.py \
    --generated-path "$result_path" \
    --split-path "$split_path" \
    --reference-path "$reference_path" \
    --eval_files_extension ".flac" \
    --ref_files_extension ".flac" \


