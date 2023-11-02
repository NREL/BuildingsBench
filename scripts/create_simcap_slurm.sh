#!/bin/bash

#SBATCH --account=foundation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=29
#SBATCH --partition=gpu
#SBATCH --mem=100gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zli4@nrel.gov


source /nopt/nrel/apps/210929a/myenv.2110041605

module purge
module load conda
module load cuda/11.7.0
module load openmpi


conda deactivate
conda activate /projects/foundation/pemami/conda/foundation

export BUILDINGS_BENCH=/projects/foundation/eulp/v1.1.0/BuildingsBench

srun python3 create_simcap_dataset.py --task get_BERT_embeddings --worker_id "$1" --worker_num "$2"