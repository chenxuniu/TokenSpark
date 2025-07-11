#!/bin/bash
#SBATCH --job-name=vllm-worker-fixed
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=vllm_worker_fixed_%j.out
#SBATCH --error=vllm_worker_fixed_%j.err
#SBATCH --nodelist=rpg-93-8
#SBATCH --exclusive

# 修复版vLLM分布式推理 - Worker节点 (长时间运行版)

echo "=== 修复版vLLM分布式推理 - Worker节点 (长时间运行) ==="
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
ray_address="$head_ip:$port"

echo "Worker connecting to: $ray_address"

# Test connectivity
echo ""
echo "🔍 Testing connectivity..."
if ping -c 3 $head_ip > /dev/null 2>&1; then
    echo "✅ Head node reachable"
else
    echo "❌ Head node unreachable"
    exit 1
fi

# Wait for head node to be ready
echo "⏳ Waiting for head node to be ready..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "--- Connection attempt $attempt/$max_attempts ---"
    
    if timeout 10 bash -c "</dev/tcp/$head_ip/$port" 2>/dev/null; then
        echo "✅ Head node is ready"
        break
    else
        echo "⏳ Head node not ready, waiting..."
        sleep 20
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Head node never became ready"
    exit 1
fi

# Start Ray worker
echo ""
echo "🚀 Starting Ray worker..."

ray start --address="$ray_address" \
    --num-cpus=8 \
    --num-gpus=4 \
    --disable-usage-stats \
    --object-store-memory=3000000000

if [ $? -eq 0 ]; then
    echo "✅ Ray worker connected successfully!"
else
    echo "❌ Ray worker failed to connect"
    exit 1
fi

# Verify connection
echo ""
echo "🧪 Verifying worker connection..."

python << 'EOF'
import ray
import time

try:
    print("🔗 Connecting to verify worker...")
    ray.init(address="10.100.93.6:6379", ignore_reinit_error=True)
    
    # Get cluster info
    nodes = ray.nodes()
    resources = ray.cluster_resources()
    alive_nodes = [n for n in nodes if n['Alive']]
    
    print(f"✅ Worker verification:")
    print(f"   Total nodes: {len(alive_nodes)}")
    print(f"   Total GPUs: {int(resources.get('GPU', 0))}")
    print(f"   Total CPUs: {int(resources.get('CPU', 0))}")
    
    # Check if this worker is recognized
    worker_found = False
    for node in alive_nodes:
        hostname = node.get('NodeManagerHostname', '')
        if 'rpg-93-8' in hostname:
            worker_found = True
            node_res = node.get('Resources', {})
            print(f"   This worker: {hostname} - {int(node_res.get('GPU', 0))} GPUs")
            break
    
    if worker_found:
        print("✅ Worker properly registered in cluster")
    else:
        print("⚠️  Worker not found in cluster")
    
    ray.shutdown()
    
except Exception as e:
    print(f"❌ Worker verification failed: {e}")
    import traceback
    traceback.print_exc()
EOF

if [ $? -eq 0 ]; then
    echo "✅ Worker verification passed!"
else
    echo "⚠️  Worker verification had issues"
fi

# FIXED: 长时间保持活跃状态
echo ""
echo "💤 Worker staying alive for the full job duration..."
echo "Ready for vLLM distributed inference"
echo "🔥 Worker will remain active until SLURM job ends or head node stops"

# 创建状态文件来跟踪worker状态
worker_status_file="/tmp/vllm_worker_status_$SLURM_JOB_ID"
echo "active" > "$worker_status_file"

# 函数：检查head node是否还在运行
check_head_node() {
    python -c "
import ray
import sys
try:
    ray.init(address='10.100.93.6:6379', ignore_reinit_error=True, _temp_dir='/tmp/ray_temp')
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n['Alive']]
    head_active = any('rpg-93-6' in n.get('NodeManagerHostname', '') for n in alive_nodes)
    
    if head_active:
        resources = ray.cluster_resources()
        worker_active = any('rpg-93-8' in n.get('NodeManagerHostname', '') for n in alive_nodes)
        print(f'OK - Cluster: {len(alive_nodes)} nodes, {int(resources.get(\"GPU\", 0))} GPUs, Head: {head_active}, Worker: {worker_active}')
        sys.exit(0)
    else:
        print('HEAD_DOWN - Head node is not active')
        sys.exit(1)
    
    ray.shutdown()
except Exception as e:
    print(f'ERROR - Connection failed: {e}')
    sys.exit(2)
" 2>/dev/null
}

# 监控循环 - 每30秒检查一次，直到SLURM任务结束
check_count=0
max_inactive_checks=10  # 最多允许10次连续失败

while [ -f "$worker_status_file" ]; do
    check_count=$((check_count + 1))
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 每分钟显示一次状态（每30秒检查，每2次显示）
    if [ $((check_count % 2)) -eq 0 ]; then
        echo "--- Worker status check #$((check_count/2)) ($current_time) ---"
        
        # 检查head node状态
        status_output=$(check_head_node)
        status_code=$?
        
        if [ $status_code -eq 0 ]; then
            echo "   $status_output"
            inactive_count=0  # 重置失败计数
        elif [ $status_code -eq 1 ]; then
            echo "   $status_output"
            echo "   Head node is down, worker will continue waiting..."
            inactive_count=$((inactive_count + 1))
        else
            echo "   $status_output"
            inactive_count=$((inactive_count + 1))
        fi
        
        # 如果连续失败太多次，考虑退出
        if [ $inactive_count -ge $max_inactive_checks ]; then
            echo "   ⚠️  Head node has been inactive for too long, but worker will keep waiting..."
            # 可以选择继续等待或退出，这里选择继续等待
            # break
        fi
        
        # 显示剩余时间信息
        job_start_time=$(squeue -j $SLURM_JOB_ID -h -o %S 2>/dev/null)
        if [ -n "$job_start_time" ]; then
            echo "   💡 Worker active since job start. Use 'scancel $SLURM_JOB_ID' to stop if needed."
        fi
    fi
    
    # 检查SLURM任务是否还在运行
    if ! squeue -j $SLURM_JOB_ID >/dev/null 2>&1; then
        echo "   🛑 SLURM job $SLURM_JOB_ID no longer in queue, worker will exit"
        break
    fi
    
    # 等待30秒后继续检查
    sleep 30
done

# 清理并退出
echo ""
echo "🛑 Worker shutting down gracefully..."
echo "   Final status check before cleanup:"

# 最后一次状态检查
final_status=$(check_head_node)
echo "   $final_status"

# 清理状态文件
rm -f "$worker_status_file"

# 停止Ray worker
echo "   Stopping Ray worker..."
ray stop --force

echo "✅ Worker job completed gracefully at $(date)"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $(hostname)"
echo "   Total checks performed: $((check_count/2))"