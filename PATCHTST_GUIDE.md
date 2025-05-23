# PatchTST Project Guide

## Project Structure

```
PatchTST/
├── PatchTST_supervised/           # Supervised learning implementation
│   ├── checkpoints/              # Saved model checkpoints
│   ├── layers/                   # Neural network layer implementations
│   ├── models/                   # Model architectures
│   ├── utils/                    # Utility functions
│   ├── data_provider/           # Data loading and preprocessing
│   ├── exp/                     # Experiment configurations
│   ├── logs/                    # Training logs
│   ├── scripts/                 # Training scripts
│   │   └── PatchTST/           # Model-specific scripts
│   │       └── weather.sh      # Weather dataset training script
│   ├── Formers/                 # Transformer implementations
│   └── run_longExp.py          # Main training script
├── PatchTST_self_supervised/    # Self-supervised learning implementation
├── dataset/                     # Dataset directory
│   └── weather/                # Weather dataset
│       └── weather.csv         # Weather data file
├── myenv/                       # Virtual environment
└── pic/                        # Project images and diagrams
```

## Important Files and Dependencies

### Core Files

1. **`run_longExp.py`** (Main Entry Point)
   - Location: `PatchTST_supervised/run_longExp.py`
   - Purpose: Main script for running experiments
   - Imports:
     - `exp.exp_main.Exp_Main` for experiment setup
     - `torch` for deep learning operations
     - `numpy` for numerical operations
     - `logging` for experiment logging

2. **`exp/exp_main.py`**
   - Location: `PatchTST_supervised/exp/exp_main.py`
   - Purpose: Defines experiment setup and training/testing logic
   - Imports:
     - Model definitions from `models/`
     - Data loaders from `data_provider/`
     - Utility functions from `utils/`

3. **`models/PatchTST.py`**
   - Location: `PatchTST_supervised/models/PatchTST.py`
   - Purpose: Implements the PatchTST model architecture
   - Imports:
     - Layer definitions from `layers/`
     - Transformer components from `Formers/`

### Training Scripts

1. **`weather.sh`**
   - Location: `PatchTST_supervised/scripts/PatchTST/weather.sh`
   - Purpose: Shell script for training on weather dataset
   - Calls: `run_longExp.py` with specific parameters

### Data Processing

1. **`data_provider/data_loader.py`**
   - Location: `PatchTST_supervised/data_provider/data_loader.py`
   - Purpose: Handles data loading and preprocessing
   - Imports:
     - `pandas` for data manipulation
     - `numpy` for numerical operations
     - `torch` for tensor operations

### Model Components

1. **`layers/`**
   - Contains custom layer implementations
   - Used by `models/PatchTST.py`

2. **`Formers/`**
   - Contains transformer-related implementations
   - Used by `models/PatchTST.py`

### Utility Files

1. **`utils/`**
   - Contains helper functions and utilities
   - Used across multiple components

### File Dependencies Graph

```
run_longExp.py
    ↓
exp_main.py
    ↓
├── models/PatchTST.py
│   ├── layers/
│   └── Formers/
├── data_provider/data_loader.py
└── utils/
```

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   
   # Install requirements
   pip install -r PatchTST_supervised/requirements.txt
   ```

2. **Dataset Preparation**
   - Download datasets from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
   - Place the datasets in the `dataset/` directory
   - For weather dataset, ensure `weather.csv` is in `dataset/weather/`

## Training Process

### 1. Running Training

The project supports both supervised and self-supervised learning. Here's how to run each:

#### Supervised Learning

For weather dataset:
```bash
cd PatchTST_supervised
./scripts/PatchTST/weather.sh
```

The script will:
- Train models for different prediction lengths (96, 192, 336, 720)
- Save logs in `PatchTST_supervised/logs/LongForecasting/`
- Save model checkpoints in `PatchTST_supervised/checkpoints/`

#### Self-supervised Learning

1. Pre-training:
```bash
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```

2. Fine-tuning:
```bash
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

### 2. Monitoring Training

Training progress can be monitored through:
- Log files in `PatchTST_supervised/logs/LongForecasting/`
- Each log file is named: `PatchTST_weather_<seq_len>_<pred_len>.log`

### 3. Key Metrics

The model outputs several metrics:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Training time
- Testing time

## Making Predictions

1. **Using Trained Models**
   - Load the saved checkpoint from `checkpoints/`
   - Use the model for inference with new data

2. **Prediction Parameters**
   - Sequence length (default: 336)
   - Prediction length (options: 96, 192, 336, 720)
   - Batch size (default: 128)
   - Learning rate (default: 0.0001)

## Model Architecture

PatchTST uses:
- Patching mechanism for time series segmentation
- Channel-independent processing
- Transformer architecture with:
  - 3 encoder layers
  - 16 attention heads
  - 128 model dimensions
  - 256 feed-forward dimensions

## Troubleshooting

1. **Dataset Issues**
   - Ensure dataset is in correct location
   - Check file permissions
   - Verify CSV format

2. **Training Issues**
   - Check GPU availability
   - Monitor memory usage
   - Verify batch size compatibility

3. **Common Errors**
   - Dataset not found: Check path in script
   - CUDA out of memory: Reduce batch size
   - Import errors: Verify environment setup

## Additional Resources

- Paper: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)
- Video Overview: https://www.youtube.com/watch?v=Z3-NrohddJw
- GitHub Repository: https://github.com/yuqinie98/PatchTST 