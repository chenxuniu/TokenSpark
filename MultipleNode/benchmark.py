#!/usr/bin/env python3
"""
vLLM Distributed Inference Benchmark Tool
"""

import argparse
import json
import time
import os
import sys
from typing import Dict, List, Any
from pathlib import Path

from dataset_loader import DatasetLoader
from vllm_engine import VLLMDistributedEngine
from timestamp_tracker import TimestampTracker
from config_manager import ConfigManager

def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description="vLLM Distributed Inference Benchmark Tool")
    
    # Model parameters
    parser.add_argument("--models", type=str, required=True,
                      help="Comma-separated list of model names (e.g., 'Llama-3.1-405B,llama-3.1-8B')")
    
    # Dataset parameters
    parser.add_argument("--datasets", type=str, default="alpaca",
                      help="Comma-separated list of datasets (alpaca,dolly,longbench,humaneval)")
    parser.add_argument("--num-samples", type=int, default=1000,
                      help="Number of samples to process per dataset")
    parser.add_argument("--min-length", type=int, default=5,
                      help="Minimum prompt length in words")
    parser.add_argument("--max-length", type=int, default=100,
                      help="Maximum prompt length in words")
    
    # Batch processing parameters
    parser.add_argument("--batch-sizes", type=str, default="256",
                      help="Comma-separated list of batch sizes (e.g., '128,256,512')")
    
    # Distributed configuration parameters
    parser.add_argument("--tensor-parallel", type=str, default="8",
                      help="Comma-separated list of tensor parallel sizes (e.g., '4,8,16')")
    parser.add_argument("--pipeline-parallel", type=str, default="2",
                      help="Comma-separated list of pipeline parallel sizes (e.g., '1,2,4')")
    parser.add_argument("--concurrency", type=str, default="1",
                      help="Comma-separated list of concurrency/instances (e.g., '1,2')")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=100,
                      help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                      help="Top-p sampling parameter")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                      help="Directory to save results")
    parser.add_argument("--timestamps-dir", type=str, default="./timestamps_results",
                      help="Directory to save timestamp logs")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Parse parameter lists
    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    tp_sizes = [int(tp.strip()) for tp in args.tensor_parallel.split(",")]
    pp_sizes = [int(pp.strip()) for pp in args.pipeline_parallel.split(",")]
    concurrencies = [int(c.strip()) for c in args.concurrency.split(",")]
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamps_dir = Path(args.timestamps_dir)
    timestamps_dir.mkdir(exist_ok=True)
    
    # Initialize components
    timer = TimestampTracker(log_file_prefix=str(timestamps_dir / "benchmark_timestamps"))
    dataset_loader = DatasetLoader()
    config_manager = ConfigManager()
    
    timer.record("benchmark_start", "Benchmark suite started")
    print(f"🚀 Starting vLLM Distributed Inference Benchmark")
    print(f"   Models: {models}")
    print(f"   Datasets: {datasets}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   TP sizes: {tp_sizes}")
    print(f"   PP sizes: {pp_sizes}")
    print(f"   Concurrencies: {concurrencies}")
    
    all_results = {}
    
    try:
        for model in models:
            model_results = {}
            print(f"\n{'='*80}")
            print(f"🔄 Benchmarking Model: {model}")
            print(f"{'='*80}")
            
            # Validate model path
            model_path = os.path.join(os.path.expanduser("~/models"), model)
            if not os.path.exists(model_path):
                print(f"❌ Model path does not exist: {model_path}")
                continue
            
            for dataset_name in datasets:
                dataset_results = {}
                print(f"\n📚 Loading dataset: {dataset_name}")
                
                # Load dataset
                prompts = dataset_loader.load_dataset(
                    dataset_name, 
                    num_samples=args.num_samples,
                    min_length=args.min_length,
                    max_length=args.max_length
                )
                
                if not prompts:
                    print(f"❌ Failed to load dataset: {dataset_name}")
                    continue
                
                print(f"✅ Loaded {len(prompts)} prompts from {dataset_name}")
                
                for tp_size in tp_sizes:
                    for pp_size in pp_sizes:
                        for concurrency in concurrencies:
                            for batch_size in batch_sizes:
                                config_key = f"TP{tp_size}_PP{pp_size}_C{concurrency}_B{batch_size}"
                                
                                print(f"\n🔧 Configuration: {config_key}")
                                
                                # Validate configuration
                                if not config_manager.validate_config(tp_size, pp_size, concurrency):
                                    print(f"❌ Invalid configuration: {config_key}")
                                    continue
                                
                                # Reset and setup timer for this configuration
                                timer = TimestampTracker(log_file_prefix=str(timestamps_dir / "timestamps"))
                                timer.set_log_file_name(
                                    model=model,
                                    dataset=dataset_name,
                                    num_prompts=len(prompts),
                                    tp=tp_size,
                                    pp=pp_size,
                                    conc=concurrency,
                                    batch_size=batch_size
                                )
                                timer.record("benchmark_config_start", f"Starting test for {config_key}")

                                # Create engine configuration
                                engine_config = {
                                    'model_path': model_path,
                                    'tensor_parallel_size': tp_size,
                                    'pipeline_parallel_size': pp_size,
                                    'concurrency': concurrency,
                                    'batch_size': batch_size,
                                    'max_tokens': args.max_tokens,
                                    'temperature': args.temperature,
                                    'top_p': args.top_p,
                                    'verbose': args.verbose
                                }
                                
                                # Run benchmark
                                result = None
                                try:
                                    engine = VLLMDistributedEngine(engine_config, timer)
                                    result = engine.run_benchmark(prompts)
                                    
                                    if result:
                                        result['config'] = config_key
                                        result['model'] = model
                                        result['dataset'] = dataset_name
                                        dataset_results[config_key] = result
                                        print(f"✅ Completed: {config_key}")
                                    else:
                                        print(f"❌ Failed: {config_key}")
                                        
                                except Exception as e:
                                    print(f"❌ Error in {config_key}: {e}")
                                finally:
                                    if result:
                                        # Save individual result file
                                        model_name_sanitized = model.replace('/', '_')
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        results_file = output_dir / (
                                            f"vllm_results_{model_name_sanitized}_{dataset_name}_"
                                            f"{len(prompts)}p_tp{tp_size}_pp{pp_size}_c{concurrency}_"
                                            f"b{batch_size}_{timestamp}.json"
                                        )
                                        with open(results_file, 'w', encoding='utf-8') as f:
                                            json.dump(result, f, indent=2, ensure_ascii=False)
                                        print(f"📁 Results for {config_key} saved to: {results_file}")
                                        
                                    timer.record("benchmark_config_end", f"Finished test for {config_key}")
                                    timer.save_to_file() # Ensure final timer data is saved
                
                if dataset_results:
                    model_results[dataset_name] = dataset_results
            
            if model_results:
                all_results[model] = model_results
        
        # Save final results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_summary_file = output_dir / f"vllm_benchmark_summary_{timestamp}.json"
        
        with open(final_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        timer.record("benchmark_complete", f"Benchmark completed, summary saved to {final_summary_file}")
        
        # Print summary
        print_benchmark_summary(all_results)
        
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📁 Summary report saved to: {final_summary_file}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        timer.record("benchmark_end", "Benchmark suite ended")
        timer.print_summary()

def print_benchmark_summary(results: Dict[str, Any]):
    """Print benchmark summary"""
    print(f"\n📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    total_tests = 0
    successful_tests = 0
    
    for model, model_results in results.items():
        print(f"\n🤖 Model: {model}")
        for dataset, dataset_results in model_results.items():
            print(f"  📚 Dataset: {dataset}")
            for config, result in dataset_results.items():
                total_tests += 1
                if result and 'performance_metrics' in result:
                    successful_tests += 1
                    metrics = result['performance_metrics']
                    throughput = metrics.get('overall_throughput_tokens_per_second', 0)
                    print(f"    ✅ {config}: {throughput:.2f} tokens/sec")
                else:
                    print(f"    ❌ {config}: Failed")
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n📈 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()