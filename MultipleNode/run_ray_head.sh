#!/bin/bash
#SBATCH --job-name=vllm-head-fixed
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_ray_%j.out
#SBATCH --error=errs/vllm_ray_%j.err
#SBATCH --nodelist=rpg-93-6
#SBATCH --exclusive

# Fixed version vLLM distributed inference - Head node

echo "=== vLLM Inference on Ray- Head Node ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Setup environment
source ~/.bashrc
cd $HOME/uv && source vllm_benchmark/bin/activate
cd $HOME/ray_vllm_benchmark

#Clear the cache
echo "🧹 Cleaning compilation caches..."
rm -rf ~/.cache/vllm/torch_compile_cache
rm -rf ~/.cache/torch/compile_cache  
rm -rf ~/.cache/triton
echo "✅ Cache cleanup completed"

# Check CUDA environment
echo "🔧 Checking CUDA environment..."
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# Ensure CUDA_VISIBLE_DEVICES is set correctly
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    echo "Set CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
fi



# Clean up existing Ray processes
echo "🧹 Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
sleep 5

# Configuration
head_ip="10.100.93.6"
port=6379

echo "Starting Ray HEAD on $head_ip:$port"

# Start Ray head
ray start --head \
    --node-ip-address="$head_ip" \
    --port=$port \
    --num-cpus=8 \
    --num-gpus=4 \
    --disable-usage-stats \
    --object-store-memory=3000000000

echo "✅ Ray Head started"

# Wait for worker
echo "🕐 Waiting for worker node to connect..."
echo "   Submit worker: sbatch run_vllm_worker.sh"

sleep 90

# Check cluster status
echo ""
echo "=== Cluster Status Check ==="
ray status

# Run the Python script
echo ""
echo "🚀 Running Distributed Inference Python Script"
echo "=" * 60

# Choose Llama-3-70B， Llama-3.1-405B，ds-R1 or llama-3.1-8B for --models
# Choose alpaca, dolly, longbench or humaneval  for --datasets
# Choose 256, 512, 1024 for --batch-sizes
# Choose 4, 8, 16 for --tensor-parallel
# Choose 1, 2, 4 for --pipeline-parallel
# Choose 1, 2, 4 for --concurrency
# Choose 512, 1024, 2048 for --max-tokens
# Choose 1000, 2000, 5000 for --num-samples

python benchmark.py \
        --models "Llama-3.1-405B" \
        --datasets "alpaca" \
        --batch-sizes "256" \
        --tensor-parallel "8" \
        --pipeline-parallel "2" \
        --concurrency "1" \
        --num-samples 12800 \
        --max-tokens 2048

# Keep alive for additional testing
echo ""
echo "💤 Keeping cluster alive for 2 minutes..."
sleep 120

# Cleanup
echo "🛑 Stopping Ray cluster"
ray stop --force

echo "=== vllm distributed inference on Ray Cluster completed ==="