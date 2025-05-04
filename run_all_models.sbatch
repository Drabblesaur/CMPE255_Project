#!/bin/bash
#SBATCH --job-name=news-bias-all-models
#SBATCH --output=logs/model_%j.out
#SBATCH --error=logs/model_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:all
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --partition=standard

# List of models to try
MODELS=("bert-base-uncased" "hkunlp/instructor-xl" "sentence-transformers/allenai-specter" "all-mpnet-base-v2")

source ~/.venv/bin/activate

# Make sure logs directory exists
mkdir -p logs

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running model: $MODEL_NAME"
    srun uv run python train.py --model_name "$MODEL_NAME"
done 