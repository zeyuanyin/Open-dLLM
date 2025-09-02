#!/bin/bash


MODEL_PATH="fredzzp/open-dcoder-0.5B"
TEMPERATURE=0.6
STEPS=64
ALG="p2"


python eval_infill.py \
    --model_path "$MODEL_PATH" \
    --task humaneval_infill \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --alg "$ALG"

python eval_infill.py \
    --model_path "$MODEL_PATH" \
    --task santacoder-fim \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --alg "$ALG"
