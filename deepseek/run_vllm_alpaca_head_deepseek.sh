#!/bin/bash
#SBATCH --job-name=vllm-head-fixed
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=vllm_deepseek_%j.out
#SBATCH --error=vllm_deepseek_%j.err
#SBATCH --nodelist=rpg-93-6
#SBATCH --exclusive

# 修复版vLLM分布式推理 - Head节点

echo "=== 修复版vLLM分布式推理 - Head节点 ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Setup environment
source ~/.bashrc
cd $HOME/uv && source vllm_benchmark/bin/activate
cd $HOME/codes/ray_vllm_405B_alpaca

#Clear the cache
echo "🧹 Cleaning compilation caches..."
rm -rf ~/.cache/vllm/torch_compile_cache
rm -rf ~/.cache/torch/compile_cache  
rm -rf ~/.cache/triton
echo "✅ Cache cleanup completed"

# 检查CUDA环境
echo "🔧 Checking CUDA environment..."
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# 确保CUDA_VISIBLE_DEVICES正确设置
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
echo "🚀 Running vLLM Distributed Inference Python Script"
echo "=" * 60

python vllm_alpaca_deepseek.py

# Keep alive for additional testing
echo ""
echo "💤 Keeping cluster alive for 2 minutes..."
sleep 120

# Cleanup
echo "🛑 Stopping Ray cluster"
ray stop --force

echo "=== vLLM distributed inference completed ==="