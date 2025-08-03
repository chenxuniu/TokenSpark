#!/bin/bash

# Batch submit Ray cluster jobs script
# Usage: ./submit_ray_jobs.sh

echo "Starting to submit Ray cluster jobs..."

# Submit head node job
echo "Submitting head node job..."
sbatch run_ray_head_node-93-6.sh
if [ $? -eq 0 ]; then
    echo "✓ Head node job submitted successfully"
else
    echo "✗ Head node job submission failed"
    exit 1
fi

# Wait a moment to ensure head node starts first
sleep 2

# Submit worker node jobs
worker_nodes=(5 7 8)
echo "Submitting worker node jobs..."

for node in "${worker_nodes[@]}"; do
    script_name="run_ray_worker_93-${node}.sh"
    echo "Submitting $script_name..."
    sbatch $script_name
    if [ $? -eq 0 ]; then
        echo "✓ Worker node 93-${node} job submitted successfully"
    else
        echo "✗ Worker node 93-${node} job submission failed"
    fi
    sleep 1  # Small interval between submissions
done

echo "All jobs submitted!"
echo "Use 'squeue -u $USER' to check job status"