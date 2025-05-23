#!/bin/bash

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "Found $GPU_COUNT GPU(s):"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
        export CUDA_VISIBLE_DEVICES=0
        echo "Using GPU: CUDA_VISIBLE_DEVICES=0"
    else
        echo "No NVIDIA GPU found. Running on CPU."
        export CUDA_VISIBLE_DEVICES=""
    fi
else
    echo "nvidia-smi not found. Running on CPU."
    export CUDA_VISIBLE_DEVICES=""
fi

# Check if dataset directory exists
if [ ! -d "/app/dataset/weather" ]; then
    echo "Error: Dataset directory not found at /app/dataset/weather"
    echo "Please mount the dataset directory when running the container"
    exit 1
fi

# Check if weather.csv exists
if [ ! -f "/app/dataset/weather/weather.csv" ]; then
    echo "Error: weather.csv not found in /app/dataset/weather"
    echo "Please ensure the dataset is properly mounted"
    exit 1
fi

# Execute the training script
cd /app/PatchTST_supervised
exec ./scripts/PatchTST/weather.sh 