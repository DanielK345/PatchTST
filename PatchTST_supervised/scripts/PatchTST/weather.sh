#!/bin/bash

# Detect OS and set project root accordingly
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Linux
    PROJECT_ROOT="$HOME/PatchTST"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    PROJECT_ROOT="/mnt/d/PKDUY/AI-ML/PatchTST"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Detect GPU and set CUDA device
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "Found $GPU_COUNT GPU(s):"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
        export CUDA_VISIBLE_DEVICES=0  # Use first GPU by default
        echo "Using GPU: CUDA_VISIBLE_DEVICES=0"
    else
        echo "No NVIDIA GPU found. Running on CPU."
        export CUDA_VISIBLE_DEVICES=""
    fi
else
    echo "nvidia-smi not found. Running on CPU."
    export CUDA_VISIBLE_DEVICES=""
fi

SUPERVISED_DIR="$PROJECT_ROOT/PatchTST_supervised"
DATASET_DIR="$PROJECT_ROOT/dataset/weather"

echo "OS detected: $OSTYPE"
echo "Project root: $PROJECT_ROOT"
echo "Supervised directory: $SUPERVISED_DIR"
echo "Dataset directory: $DATASET_DIR"

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found at $DATASET_DIR"
    echo "Please download the datasets from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy"
    echo "And place them in the dataset directory"
    exit 1
fi

# Check if weather.csv exists
if [ ! -f "$DATASET_DIR/weather.csv" ]; then
    echo "Error: weather.csv not found in $DATASET_DIR"
    echo "Please download the weather dataset and place it in the dataset directory"
    exit 1
fi

# Create logs directory if it doesn't exist
if [ ! -d "$SUPERVISED_DIR/logs" ]; then
    echo "Creating logs directory..."
    mkdir -p "$SUPERVISED_DIR/logs"
fi

if [ ! -d "$SUPERVISED_DIR/logs/LongForecasting" ]; then
    echo "Creating LongForecasting logs directory..."
    mkdir -p "$SUPERVISED_DIR/logs/LongForecasting"
fi

echo "Starting PatchTST training for weather dataset..."

seq_len=336
model_name=PatchTST
root_path_name="$DATASET_DIR/"
data_path_name="weather.csv"
model_id_name="weather"
data_name="custom"
random_seed=2021
pred_lens=(96 192 336 720)

echo "Configuration:"
echo "Sequence length: $seq_len"
echo "Model: $model_name"
echo "Random seed: $random_seed"
echo "Dataset path: $root_path_name"

for pred_len in "${pred_lens[@]}"
do
    echo "Training with prediction length: $pred_len"
    echo "Running experiment with following parameters:"
    echo "- Model: $model_name"
    echo "- Dataset: $data_name"
    echo "- Features: M"
    echo "- Sequence length: $seq_len"
    echo "- Prediction length: $pred_len"
    echo "- Batch size: 128"
    echo "- Learning rate: 0.0001"
    
    cd "$SUPERVISED_DIR"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name"_"$seq_len"_"$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 > logs/LongForecasting/$model_name"_"$model_id_name"_"$seq_len"_"$pred_len.log
    
    echo "Training completed. Logs saved to logs/LongForecasting/$model_name"_"$model_id_name"_"$seq_len"_"$pred_len.log"
done 