# PatchTST Docker Setup

This guide explains how to run PatchTST using Docker on any server with NVIDIA GPU support.

## Prerequisites

1. Install Docker: https://docs.docker.com/get-docker/
2. Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. Download the weather dataset from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

## Building the Docker Image

```bash
# Build the Docker image
docker build -t patchtst .
```

## Running the Container

### Basic Usage

```bash
# Run with GPU support
docker run --gpus all \
    -v /path/to/your/dataset:/app/dataset \
    patchtst
```

### Advanced Usage

```bash
# Run with specific GPU
docker run --gpus '"device=0"' \
    -v /path/to/your/dataset:/app/dataset \
    patchtst

# Run with CPU only
docker run \
    -v /path/to/your/dataset:/app/dataset \
    patchtst
```

## Directory Structure

The container expects the following directory structure:
```
/app/
├── dataset/
│   └── weather/
│       └── weather.csv
├── PatchTST_supervised/
│   ├── logs/
│   │   └── LongForecasting/
│   └── checkpoints/
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Controls which GPU to use (set automatically by the container)
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered

## Troubleshooting

1. **GPU not detected**
   - Ensure NVIDIA drivers are installed on the host
   - Verify NVIDIA Container Toolkit is installed
   - Check GPU visibility with `nvidia-smi` on the host

2. **Dataset not found**
   - Verify the dataset path is correct
   - Check the volume mount in the docker run command
   - Ensure weather.csv is in the correct location

3. **Permission issues**
   - Run with appropriate user permissions
   - Check file permissions in the mounted dataset directory

## Monitoring

- Training logs are saved to `/app/PatchTST_supervised/logs/LongForecasting/`
- Model checkpoints are saved to `/app/PatchTST_supervised/checkpoints/`

To view logs in real-time:
```bash
docker logs -f <container_id>
``` 