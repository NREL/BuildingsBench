#!/bin/sh

echo "submitting $1 jobs"

for i in $(eval echo {1..$1}); do
    sbatch --job-name "simcap$i" --output "/projects/foundation/zhaonan/logs/simcap_$i.txt" --error "/projects/foundation/zhaonan/logs/simcap_$i.error" create_simcap_slurm.sh "$i" "$1"
done