# TokenSpark Benchmark

A comprehensive benchmark suite for measuring energy consumption of large language models using Ray and vLLM across distributed GPU clusters.

## Overview

This benchmark suite evaluates the energy efficiency and performance characteristics of different large language models in distributed inference scenarios. It provides detailed energy consumption metrics, performance analysis, and scalability measurements across multi-node GPU clusters.

## Project Structure

```
ray-vllm-energy-benchmark/
├── llama-405b/                 # Llama 3.1 405B benchmarks
│   ├── run_vllm_alpaca_head_405B.sh
│   ├── run_vllm_worker_93-*.sh
│   ├── vllm_alpaca_405B.py
│   └── README.md
├── deepseek/                   # DeepSeek R1 model benchmarks
│   ├── run_vllm_alpaca_head_deepseek.sh
│   ├── run_vllm_worker_93-*.sh
│   ├── vllm_alpaca_deepseek.py
│   └── README.md
├── chunked-benchmark/          # Automated multi-model benchmarks
│   ├── benchmark.py            # Main benchmark entry point
│   ├── dataset_loader.py       # Multi-dataset support
│   ├── vllm_engine.py         # Distributed inference engine
│   ├── config_manager.py      # Configuration validation
│   ├── timestamp_tracker.py   # Performance tracking
│   ├── run_ray_head_node-93-6.sh
│   ├── run_ray_worker_93-*.sh
│   ├── submit_ray_jobs.sh
│   └── README.md
```

## Features

- **Multi-Model Support**: Benchmarks for Llama 405B, 70B, 8B, DeepSeek R1, and other large language models
- **Multi-Dataset Integration**: Alpaca, Dolly 15K, LongBench, HumanEval for comprehensive evaluation
- **Distributed Inference**: Ray-based multi-node GPU cluster deployment
- **Automated Configuration**: Smart TP/PP validation and optimization recommendations
- **Energy Monitoring**: Real-time power consumption tracking across GPUs
- **Performance Analysis**: Comprehensive metrics including throughput, latency, and tokens/sec
- **Scalability Testing**: Various tensor parallel and pipeline parallel configurations
- **Dataset Integration**: Realistic workload simulation with instruction datasets
- **Batch Processing**: Configurable batch sizes for optimal performance
- **Comprehensive Logging**: Detailed timestamps and performance tracking

## Prerequisites

### Hardware Requirements
- Multi-node GPU cluster (H100/A100 recommended)
- Minimum 10 GPUs for 405B model testing
- High-speed interconnect (InfiniBand recommended)
- SLURM job scheduler

### Software Requirements
```bash
# Core dependencies
python >= 3.9
ray >= 2.22.0
vllm >= 0.4.0
torch >= 2.0.0
transformers
datasets

# Energy monitoring
nvidia-ml-py
psutil
```

## Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/chenxuniu/TokenSpark.git
cd 405B

# Setup virtual environment
python -m venv vllm_benchmark
source vllm_benchmark/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Cluster
```bash
# Update node IPs in configuration files
# Edit head node IP in all worker scripts
vim llama-405b/run_vllm_worker_*.sh
```

### 3. Run Benchmarks

#### Llama 405B Benchmark
```bash
cd 405B/

# Start head node
sbatch run_vllm_alpaca_head_405B.sh

# Start worker nodes (in separate terminals)
sbatch run_vllm_worker_93-4.sh
sbatch run_vllm_worker_93-5.sh
sbatch run_vllm_worker_93-8.sh
```

#### Benchmark (Automated)
```bash
cd llm_energy_benchmark/

# Submit Ray cluster jobs
./submit_ray_jobs.sh

# Run comprehensive benchmark
python benchmark.py \
    --models "Llama-3.1-405B,DeepSeek-R1,Llama-3.1-8B" \
    --datasets "alpaca,dolly,humaneval" \
    --batch-sizes "256,512,1024" \
    --tensor-parallel "4,8,16" \
    --pipeline-parallel "1,2" \
    --concurrency "1,2" \
    --num-samples 1000 \
    --max-tokens 512 \
    --verbose
```

## Benchmark Configurations

### Llama 405B
- **Tensor Parallel**: 8-way (TP=8)
- **Pipeline Parallel**: 2-way (PP=2)
- **Quantization**: FP8
- **Memory Usage**: ~1.5TB across 16 H100 GPUs
- **Dataset**: Alpaca instructions (52K total)

### DeepSeek R1
- **Model Size**: ~70B parameters
- **Tensor Parallel**: 8-way (TP=8) or 4-way (TP=4)
- **Pipeline Parallel**: 2-way (PP=2)
- **Memory Usage**: ~600GB across 16 H100 GPUs
- **Context Length**: 4096 tokens
- **Dataset**: Alpaca instructions (500 samples)

### Chunked Benchmark (Automated)
- **Multi-Model Support**: Llama 405B, 70B, DeepSeek R1, 8B models
- **Multi-Dataset**: Alpaca, Dolly 15K, LongBench, HumanEval
- **Automated Configuration**: Smart TP/PP optimization
- **Batch Processing**: Configurable batch sizes (64-1024)
- **Energy Tracking**: Integrated power monitoring
- **Comprehensive Analysis**: Detailed performance metrics
| Model | TP | PP | GPUs | Memory | Batch Size | Context |
|-------|----|----|------|--------|------------|---------|
| Llama 405B | 8 | 2 | 16 | 1.5TB | 256 | 2048 |
| Llama 405B | 4 | 4 | 16 | 1.5TB | 512 | 2048 |
| Llama 405B | 16 | 1 | 16 | 1.5TB | 1024 | 2048 |
| DeepSeek R1 | 8 | 2 | 16 | 600GB | 256 | 4096 |
| DeepSeek R1 | 4 | 2 | 8 | 400GB | 512 | 4096 |

## Energy Monitoring

### Metrics Collected
- **Power Consumption**: Per-GPU power draw (Watts)
- **Energy Usage**: Total energy consumption (kWh)
- **GPU Utilization**: Compute and memory utilization
- **Temperature**: GPU temperature monitoring
- **Performance**: Tokens/sec, latency, throughput

## Results Analysis

### Performance Metrics
- **Throughput**: Tokens per second
- **Latency**: Time to first token
- **Efficiency**: Tokens per second per GPU
- **Scalability**: Performance vs. GPU count

### Energy Metrics
- **Power per Token**: Watts per generated token
- **Energy Efficiency**: Tokens per kWh
- **Total Energy Cost**: kWh for complete benchmark
- **Power Profile**: Power consumption over time

## Usage Examples

## Usage Examples

### Basic Individual Benchmarks

#### Llama 405B Benchmark
```bash
cd 405B/
sbatch run_vllm_alpaca_head_405B.sh
# Start worker nodes
sbatch run_vllm_worker_93-4.sh
sbatch run_vllm_worker_93-5.sh
sbatch run_vllm_worker_93-8.sh
# Start workers: run_vllm_worker_93-*.sh
```

#### DeepSeek R1 Benchmark
```bash
cd deepseek/
sbatch run_vllm_alpaca_head_deepseek.sh
sbatch run_vllm_worker_93-4.sh
sbatch run_vllm_worker_93-5.sh
sbatch run_vllm_worker_93-8.sh
# Start workers: run_vllm_worker_93-*.sh
```

#### Comprehensive Automated Benchmark
```bash
cd llm_energy_benchmark/
./submit_ray_jobs.sh  # Start Ray cluster

# Multi-model comparison
python benchmark.py \
    --models "Llama-3.1-405B,DeepSeek-R1,Llama-3.1-8B" \
    --datasets "alpaca,dolly" \
    --batch-sizes "256,512" \
    --tensor-parallel "4,8" \
    --pipeline-parallel "1,2" \
    --num-samples 1000
```

### Advanced Configuration Examples

#### Large Scale Testing
```bash
# Test multiple configurations for 405B model
python benchmark.py \
    --models "Llama-3.1-405B" \
    --datasets "alpaca,longbench,humaneval" \
    --batch-sizes "128,256,512,1024" \
    --tensor-parallel "8,16" \
    --pipeline-parallel "1,2,4" \
    --concurrency "1,2" \
    --num-samples 5000 \
    --max-tokens 1024 \
    --verbose
```

#### Quick Performance Test
```bash
# Fast test for development
python benchmark.py \
    --models "Llama-3.1-8B" \
    --datasets "alpaca" \
    --batch-sizes "512" \
    --tensor-parallel "2" \
    --pipeline-parallel "1" \
    --concurrency "1" \
    --num-samples 100 \
    --max-tokens 256
```

#### Coding Benchmark
```bash
# HumanEval coding benchmark
python benchmark.py \
    --models "DeepSeek-R1,Llama-3.1-405B" \
    --datasets "humaneval" \
    --batch-sizes "256" \
    --tensor-parallel "8" \
    --pipeline-parallel "2" \
    --num-samples 500 \
    --max-tokens 512 \
    --temperature 0.1
```

### Custom Configuration

#### Llama 405B Custom Settings
```python
# Modify vllm_alpaca_405B.py
tensor_parallel_size = 4    # Change TP size
pipeline_parallel_size = 4  # Change PP size
batch_size = 512           # Adjust batch size
num_samples = 1000         # Set sample count
quantization = "fp8"       # Enable FP8 quantization
```

#### Chunked Benchmark Configuration
```python
# Modify benchmark.py or use command line arguments
python benchmark.py \
    --models "Llama-3.1-405B,DeepSeek-R1" \
    --datasets "alpaca,longbench" \
    --batch-sizes "128,256,512" \
    --tensor-parallel "8" \
    --pipeline-parallel "2" \
    --concurrency "1" \
    --num-samples 5000 \
    --max-tokens 1024 \
    --output-dir "./results" \
    --timestamps-dir "./timing_logs"
```

#### Quick Single Model Test
```bash
# Test single configuration quickly
python benchmark.py \
    --models "Llama-3.1-8B" \
    --datasets "alpaca" \
    --batch-sizes "512" \
    --tensor-parallel "2" \
    --pipeline-parallel "1" \
    --concurrency "1" \
    --num-samples 100
```

### Energy Analysis
```bash
# Generate energy report
python energy-monitoring/power-analysis.py \
    --input energy_log.csv \
    --output energy_report.html \
    --model llama-405b
```

## Output Files

### Generated Results
- `vllm_405b_alpaca_results_TIMESTAMP.json`: Llama 405B benchmark results
- `vllm_deepseek_r1_alpaca_results_TIMESTAMP.json`: DeepSeek R1 benchmark results
- `vllm_results_MODEL_DATASET_CONFIG_TIMESTAMP.json`: Individual test results
- `vllm_benchmark_summary_TIMESTAMP.json`: Comprehensive benchmark summary
- `vllm_*_timestamps_TIMESTAMP.json`: Detailed timing analysis
- `energy_log_TIMESTAMP.csv`: Power consumption data
- `performance_metrics_TIMESTAMP.json`: Performance statistics

### Log Files
- `vllm_405B_JOBID.out`: Llama 405B head node execution logs
- `vllm_deepseek_JOBID.out`: DeepSeek head node execution logs
- `vllm_ray_JOBID.out`: Chunked benchmark head node logs
- `vllm_worker*_JOBID.out`: Worker node logs
- `slurm_monitoring.log`: SLURM job monitoring

## Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce memory usage
gpu_memory_utilization = 0.80  # from 0.90
max_model_len = 512           # from 1024
batch_size = 128              # from 256
```

**Connection Issues**
```bash
# Check Ray cluster status
ray status

# Verify node connectivity
ping <head-node-ip>

# Check SLURM jobs
squeue -u $USER
```

#### Chunked Benchmark Issues
```bash
# Configuration validation errors
python -c "
from config_manager import ConfigManager
cm = ConfigManager()
cm.print_cluster_info()
cm.print_recommended_configs('Llama-3.1-405B')
"

# Dataset loading issues
python -c "
from dataset_loader import DatasetLoader
dl = DatasetLoader()
print(dl.get_dataset_info())
prompts = dl.load_dataset('alpaca', num_samples=10)
print(f'Loaded {len(prompts)} prompts')
"

# Automated job submission
chmod +x submit_ray_jobs.sh
./submit_ray_jobs.sh
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-benchmark`)
3. Commit changes (`git commit -am 'Add new benchmark'`)
4. Push to branch (`git push origin feature/new-benchmark`)
5. Create Pull Request

## License

<!-- MIT License - see LICENSE file for details -->

<!-- ## Citation

```bibtex
@misc{ray-vllm-energy-benchmark,
  title={Ray vLLM Energy Benchmark Suite},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo/ray-vllm-energy-benchmark}}
}
``` -->

## Acknowledgments

- Ray Team for distributed computing framework
- vLLM Team for efficient LLM serving
- Hugging Face for model hosting and datasets
- NVIDIA for GPU computing infrastructure