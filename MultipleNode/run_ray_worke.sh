#!/bin/bash
#SBATCH --job-name=vllm-worker4-fixed
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=worker_logs/vllm_worker4_fixed_%j.out
#SBATCH --error=worker_errs/vllm_worker4_fixed_%j.err
#SBATCH --nodelist=rpg-93-4 
#SBATCH --exclusive

echo "=== vLLM Inference on Ray - Worker Node ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Setup environment
source ~/.bashrc
cd $HOME/uv && source vllm_benchmark/bin/activate
cd $HOME/ray_vllm_benchmark

# Clean vLLM cache
echo "🧹 Cleaning vLLM cache..."
rm -rf ~/.cache/vllm/torch_compile_cache
rm -rf ~/.cache/vllm/
rm -rf ~/.cache/torch/compile_cache  
rm -rf ~/.cache/triton
echo "✅ Cache cleanup completed"

# check CUDA environment
echo "🔧 Checking CUDA environment..."
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
nvidia-smi --query-gpu=index,name --format=csv,noheader

# ensure CUDA_VISIBLE_DEVICES is set correctly
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

echo "Worker4 connecting to: $ray_address"

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
echo "🚀 Starting Ray worker4..."

ray start --address="$ray_address" \
    --num-cpus=8 \
    --num-gpus=4 \
    --disable-usage-stats \
    --object-store-memory=3000000000

if [ $? -eq 0 ]; then
    echo "✅ Ray worker4 connected successfully!"
else
    echo "❌ Ray worker4 failed to connect"
    exit 1
fi

# Verify connection
echo ""
echo "🧪 Verifying worker4 connection..."

python << 'EOF'
import ray
import time

try:
    print("🔗 Connecting to verify worker4...")
    ray.init(address="10.100.93.6:6379", ignore_reinit_error=True)
    
    # Get cluster info
    nodes = ray.nodes()
    resources = ray.cluster_resources()
    alive_nodes = [n for n in nodes if n['Alive']]
    
    print(f"✅ Worker4 verification:")
    print(f"   Total nodes: {len(alive_nodes)}")
    print(f"   Total GPUs: {int(resources.get('GPU', 0))}")
    print(f"   Total CPUs: {int(resources.get('CPU', 0))}")
    
    # List all nodes in cluster
    print(f"   Active nodes in cluster:")
    for i, node in enumerate(alive_nodes):
        hostname = node.get('NodeManagerHostname', 'unknown')
        node_res = node.get('Resources', {})
        gpu_count = int(node_res.get('GPU', 0))
        cpu_count = int(node_res.get('CPU', 0))
        
        if 'rpg-93-6' in hostname:
            node_type = "(HEAD)"
        elif 'rpg-93-8' in hostname:
            node_type = "(WORKER2)"
        elif 'rpg-93-5' in hostname:
            node_type = "(WORKER3)"
        elif 'rpg-93-4' in hostname:
            node_type = "(WORKER4 - THIS NODE)"
        else:
            node_type = "(UNKNOWN)"
        
        print(f"     Node {i+1}: {hostname} {node_type} - {gpu_count} GPUs, {cpu_count} CPUs")
    
    # Check if this worker is recognized
    worker_found = False
    for node in alive_nodes:
        hostname = node.get('NodeManagerHostname', '')
        if 'rpg-93-4' in hostname:
            worker_found = True
            node_res = node.get('Resources', {})
            print(f"\n   ✅ This worker4: {hostname} - {int(node_res.get('GPU', 0))} GPUs")
            break
    
    if worker_found:
        print("✅ Worker4 properly registered in cluster")
    else:
        print("⚠️  Worker4 not found in cluster")
    
    ray.shutdown()
    
except Exception as e:
    print(f"❌ Worker4 verification failed: {e}")
    import traceback
    traceback.print_exc()
EOF

if [ $? -eq 0 ]; then
    echo "✅ Worker4 verification passed!"
else
    echo "⚠️  Worker4 verification had issues"
fi

# FIXED: Keep alive for extended period
echo ""
echo "💤 Worker4 staying alive for the full job duration..."
echo "Ready for vLLM distributed inference"
echo "🔥 Worker4 will remain active until SLURM job ends or head node stops"

# Create status file to track worker4 status
worker4_status_file="/tmp/vllm_worker4_status_$SLURM_JOB_ID"
echo "active" > "$worker4_status_file"

# Function: Check if head node is still running and display cluster status
check_head_node() {
    python -c "
import ray
import sys
try:
    ray.init(address='10.100.93.6:6379', ignore_reinit_error=True, _temp_dir='/tmp/ray_temp_worker4')
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n['Alive']]
    head_active = any('rpg-93-6' in n.get('NodeManagerHostname', '') for n in alive_nodes)
    
    if head_active:
        resources = ray.cluster_resources()
        
        # Count node status
        node_status = {
            'head': False,
            'worker2': False, 
            'worker3': False,
            'worker4': False
        }
        
        for node in alive_nodes:
            hostname = node.get('NodeManagerHostname', '')
            if 'rpg-93-6' in hostname:
                node_status['head'] = True
            elif 'rpg-93-8' in hostname:
                node_status['worker2'] = True
            elif 'rpg-93-5' in hostname:
                node_status['worker3'] = True
            elif 'rpg-93-4' in hostname:
                node_status['worker4'] = True
        
        # Generate status summary
        active_workers = sum([node_status['worker2'], node_status['worker3'], node_status['worker4']])
        total_gpus = int(resources.get('GPU', 0))
        
        status_summary = f'OK - Cluster: {len(alive_nodes)} nodes, {total_gpus} GPUs total'
        node_summary = f'Head:{\"✓\" if node_status[\"head\"] else \"✗\"}, W2:{\"✓\" if node_status[\"worker2\"] else \"✗\"}, W3:{\"✓\" if node_status[\"worker3\"] else \"✗\"}, W4:{\"✓\" if node_status[\"worker4\"] else \"✗\"}'
        
        print(f'{status_summary} ({node_summary})')
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

# Monitoring loop - check every 30 seconds until SLURM job ends
check_count=0
max_inactive_checks=10  # Allow up to 10 consecutive failures
inactive_count=0

echo "📊 Worker4 monitoring started - checking every 30 seconds"
echo "   Use 'tail -f vllm_worker4_fixed_${SLURM_JOB_ID}.out' to monitor this log"

while [ -f "$worker4_status_file" ]; do
    check_count=$((check_count + 1))
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Display status every minute (check every 30 seconds, display every 2 checks)
    if [ $((check_count % 2)) -eq 0 ]; then
        echo ""
        echo "--- Worker4 status check #$((check_count/2)) ($current_time) ---"
        
        # Check head node status
        status_output=$(check_head_node)
        status_code=$?
        
        if [ $status_code -eq 0 ]; then
            echo "   $status_output"
            inactive_count=0  # Reset failure count
        elif [ $status_code -eq 1 ]; then
            echo "   $status_output"
            echo "   Head node is down, worker4 will continue waiting..."
            inactive_count=$((inactive_count + 1))
        else
            echo "   $status_output"
            inactive_count=$((inactive_count + 1))
        fi
        
        # If too many consecutive failures, show warning but continue waiting
        if [ $inactive_count -ge $max_inactive_checks ]; then
            echo "   ⚠️  Head node has been inactive for $(($inactive_count * 30)) seconds, but worker4 will keep waiting..."
        fi
        
        # Display job information
        if [ $((check_count/2 % 5)) -eq 0 ]; then  # Display detailed info every 5 minutes
            echo "   💡 Worker4 (rpg-93-4) has been active for $((check_count * 30 / 60)) minutes"
            echo "   💡 Use 'scancel $SLURM_JOB_ID' to stop this worker if needed"
            
            # Display SLURM job remaining time
            time_left=$(squeue -j $SLURM_JOB_ID -h -o %L 2>/dev/null)
            if [ -n "$time_left" ]; then
                echo "   ⏰ SLURM job time remaining: $time_left"
            fi
        fi
    fi
    
    # Check if SLURM job is still running
    if ! squeue -j $SLURM_JOB_ID >/dev/null 2>&1; then
        echo "   🛑 SLURM job $SLURM_JOB_ID no longer in queue, worker4 will exit"
        break
    fi
    
    # Wait 30 seconds before next check
    sleep 30
done

# Cleanup and exit
echo ""
echo "🛑 Worker4 shutting down gracefully..."
echo "   Final status check before cleanup:"

# Final status check
final_status=$(check_head_node)
echo "   $final_status"

# Clean up status file
rm -f "$worker4_status_file"

# Stop Ray worker
echo "   Stopping Ray worker4..."
ray stop --force

echo ""
echo "✅ Worker4 job completed gracefully at $(date)"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: rpg-93-4"
echo "   Total runtime: $((check_count * 30 / 60)) minutes"
echo "   Total status checks: $((check_count/2))"