#!/bin/bash
#SBATCH --job-name=news-bias-all-models
#SBATCH --output=/home/017534556/projects/255_final/logs/news_bias_%j.out
#SBATCH --error=/home/017534556/projects/255_final/logs/news_bias_%j.err
#SBATCH --partition=condo
#SBATCH --nodelist=condo8
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=saishreddy.komalla@sjsu.edu
#SBATCH --mail-type=BEGIN,FAIL,END

# List of models to try
# MODELS=("bert-base-uncased" "hkunlp/instructor-xl" "sentence-transformers/allenai-specter" "all-mpnet-base-v2")
# MODELS=("sentence-transformers/all-mpnet-base-v2")
MODELS=("microsoft/deberta-v3-large")

source /home/017534556/projects/255_final/.venv/bin/activate

# Make sure logs directory exists
mkdir -p /home/017534556/projects/255_final/logs

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running model: $MODEL_NAME"
    srun uv run python train.py --model_name "$MODEL_NAME"
done 