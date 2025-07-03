#!/bin/bash

ENGINE="vllm"
MODEL="Llama-3.2-3B"
MODEL_PATH="/mnt/REPACSS/home/tongywan/models/${MODEL}" 



BATCH_SIZES=(128 256)
NUM_SAMPLES=(64 128 256 512)
MIN_LENGTHS=(0 20 40) 
MAX_LENGTHS=(19 39 59) 
OUTPUT_TOKENS=128 

 
LOG_DIR="/mnt/REPACSS/home/tongywan/logs" 
mkdir -p $LOG_DIR

for i in "${!MIN_LENGTHS[@]}"; do
  MIN_LEN=${MIN_LENGTHS[$i]}
  MAX_LEN=${MAX_LENGTHS[$i]}
  for BATCH in "${BATCH_SIZES[@]}"; do
    for SAMPLE in "${NUM_SAMPLES[@]}"; do

      echo "Running: batch_size=$BATCH, num_samples=$SAMPLE, min_length=$MIN_LEN, max_length=$MAX_LEN"
      python llm_benchmark.py \
        --engine $ENGINE \
        --models $MODEL \
        --batch-sizes $BATCH \
        --output-tokens $OUTPUT_TOKENS \
        --num-samples $SAMPLE \
        --min-length $MIN_LEN \
        --max-length $MAX_LEN \
        > "$LOG_DIR/${MODEL}_B${BATCH}_S${SAMPLE}_L${MIN_LEN}-${MAX_LEN}.log" 2>&1

    done
  done
done
