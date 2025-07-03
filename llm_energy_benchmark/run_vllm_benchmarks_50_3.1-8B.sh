#!/bin/bash

ENGINE="vllm"
MODEL="Llama-3.1-8B"
MODEL_PATH="/mnt/REPACSS/home/tongywan/models/${MODEL}"

BATCH_SIZES=(128 256 512)
NUM_SAMPLES=(32 64 128 256 512)
MIN_LENGTHS=(0 50)
MAX_LENGTHS=(49 99)
OUTPUT_TOKENS_LIST=(256 500 512 1000 2000)

LOG_DIR="/mnt/REPACSS/home/tongywan/logs"
mkdir -p $LOG_DIR

for i in "${!MIN_LENGTHS[@]}"; do
  MIN_LEN=${MIN_LENGTHS[$i]}
  MAX_LEN=${MAX_LENGTHS[$i]}
  for BATCH in "${BATCH_SIZES[@]}"; do
    for SAMPLE in "${NUM_SAMPLES[@]}"; do
      for OUTPUT_TOKENS in "${OUTPUT_TOKENS_LIST[@]}"; do

        echo "Running: batch_size=$BATCH, num_samples=$SAMPLE, min_length=$MIN_LEN, max_length=$MAX_LEN, output_tokens=$OUTPUT_TOKENS"
        python llm_benchmark.py \
          --engine $ENGINE \
          --models $MODEL \
          --batch-sizes $BATCH \
          --output-tokens $OUTPUT_TOKENS \
          --num-samples $SAMPLE \
          --min-length $MIN_LEN \
          --max-length $MAX_LEN \
          > "$LOG_DIR/${MODEL}_B${BATCH}_S${SAMPLE}_O${OUTPUT_TOKENS}_L${MIN_LEN}-${MAX_LEN}.log" 2>&1

      done
    done
  done
done

