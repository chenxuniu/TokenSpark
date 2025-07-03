"""
vLLM 405B Distributed Inference with Alpaca Dataset - WITH DETAILED TIMESTAMPS
Cross-node inference using Alpaca instruction dataset with comprehensive timing analysis
"""

from typing import Dict, List
import os
import time
import json
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from datetime import datetime

# Timestamp tracker class
class TimestampTracker:
    def __init__(self):
        self.timestamps = {}
        self.stage_durations = {}
        self.log_file = f"vllm_405b_timestamps_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
    def record(self, stage_name: str, description: str = ""):
        """Record a timestamp for a specific stage"""
        timestamp = time.time()
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        self.timestamps[stage_name] = {
            "timestamp": timestamp,
            "readable_time": readable_time,
            "description": description
        }
        
        print(f"⏱️  [{stage_name}] {readable_time} - {description}")
        
        # Calculate duration from previous stage
        stage_keys = list(self.timestamps.keys())
        if len(stage_keys) > 1:
            prev_stage = stage_keys[-2]
            current_stage = stage_keys[-1]
            duration = timestamp - self.timestamps[prev_stage]["timestamp"]
            self.stage_durations[f"{prev_stage}_to_{current_stage}"] = duration
            print(f"   ⏳ Duration from {prev_stage}: {duration:.2f}s")
        
        # Auto-save to file
        self.save_to_file()
    
    def get_duration(self, start_stage: str, end_stage: str) -> float:
        """Get duration between two stages"""
        if start_stage in self.timestamps and end_stage in self.timestamps:
            return self.timestamps[end_stage]["timestamp"] - self.timestamps[start_stage]["timestamp"]
        return 0.0
    
    def save_to_file(self):
        """Save timestamps to JSON file"""
        try:
            data = {
                "timestamps": self.timestamps,
                "stage_durations": self.stage_durations,
                "summary": {
                    "total_stages": len(self.timestamps),
                    "start_time": list(self.timestamps.values())[0]["readable_time"] if self.timestamps else None,
                    "end_time": list(self.timestamps.values())[-1]["readable_time"] if self.timestamps else None
                }
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Failed to save timestamps: {e}")
    
    def print_summary(self):
        """Print a summary of all recorded timestamps"""
        print(f"\n📊 TIMING SUMMARY")
        print("=" * 80)
        
        for stage, info in self.timestamps.items():
            print(f"   {stage}: {info['readable_time']} - {info['description']}")
        
        print(f"\n⏱️  STAGE DURATIONS")
        print("-" * 60)
        for duration_name, duration in self.stage_durations.items():
            print(f"   {duration_name}: {duration:.2f}s ({duration/60:.1f}min)")
        
        if len(self.timestamps) >= 2:
            total_time = self.get_duration(
                list(self.timestamps.keys())[0], 
                list(self.timestamps.keys())[-1]
            )
            print(f"\n🕐 TOTAL TIME: {total_time:.2f}s ({total_time/60:.1f}min)")
        
        print(f"\n💾 Detailed timestamps saved to: {self.log_file}")

# Alpaca dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

def load_alpaca_dataset(min_length: int = 0, max_length: int = float('inf'), 
                       num_samples: int = 100, cache_dir: str = None) -> List[str]:
    """Load and filter Alpaca dataset instructions."""
    if not DATASETS_AVAILABLE:
        print("Error: Hugging Face datasets library is not available. Cannot load Alpaca dataset.")
        return [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the difference between supervised and unsupervised learning?"
        ]
    
    try:
        print(f"🔄 Loading Alpaca dataset...")
        alpaca_ds = load_dataset("tatsu-lab/alpaca", cache_dir=cache_dir)
        
        all_prompts = []
        
        if "train" in alpaca_ds:
            for item in alpaca_ds["train"]:
                if "instruction" in item and item["instruction"].strip():
                    instruction = item["instruction"].strip()
                    
                    # Add input context if available
                    if "input" in item and item["input"].strip():
                        instruction = f"{instruction}\n\nContext: {item['input'].strip()}"
                    
                    all_prompts.append(instruction)
        
        total_prompts = len(all_prompts)
        print(f"📊 Loaded {total_prompts} raw prompts from Alpaca dataset")
        
        # Filter by length
        if min_length > 0 or max_length < float('inf'):
            filtered_prompts = []
            for prompt in all_prompts:
                words = len(prompt.split())
                if min_length <= words <= max_length:
                    filtered_prompts.append(prompt)
            
            print(f"📋 Filtered to {len(filtered_prompts)} prompts between {min_length} and {max_length} words")
            all_prompts = filtered_prompts
        
        # Limit number of samples - prioritize first num_samples
        if num_samples < len(all_prompts):
            all_prompts = all_prompts[:num_samples]
            print(f"🎯 Using first {num_samples} prompts from Alpaca dataset (out of {total_prompts} total)")
        else:
            print(f"✅ Using all {len(all_prompts)} available prompts from Alpaca dataset")
        
        return all_prompts
    
    except Exception as e:
        print(f"❌ Error loading Alpaca dataset: {e}")
        # Fallback prompts
        return [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the difference between supervised and unsupervised learning?"
        ]

def save_results_to_json(results: List[Dict], model_name: str, 
                        total_time: float, performance_metrics: Dict, timestamps: dict):
    """Save results to JSON file with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"vllm_405b_alpaca_results_{timestamp}.json"
    
    output_data = {
        "metadata": {
            "model": model_name,
            "timestamp": timestamp,
            "total_inference_time": total_time,
            "performance_metrics": performance_metrics,
            "total_prompts": len(results),
            "detailed_timestamps": timestamps  # Include timestamp data
        },
        "results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")
    return filename

# Initialize timestamp tracker
timer = TimestampTracker()

print("🔧 Starting vLLM 405B distributed inference with Alpaca dataset...")

timer.record("script_start", "Python script execution started")

try:
    # Debug CUDA environment
    timer.record("cuda_check_start", "Starting CUDA environment check")
    print(f"🔍 CUDA environment check:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    # Check Ray version
    print(f"📦 Ray version: {ray.__version__}")
    assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"
    timer.record("cuda_check_complete", "CUDA environment check completed")
    
    # Connect to Ray cluster
    timer.record("ray_init_start", "Starting Ray cluster connection")
    ray.init(address="auto", ignore_reinit_error=True)
    timer.record("ray_init_complete", "Ray cluster connection established")
    
    # Get cluster info
    timer.record("cluster_info_start", "Gathering cluster information")
    nodes = ray.nodes()
    resources = ray.cluster_resources()
    alive_nodes = [n for n in nodes if n['Alive']]
    total_gpus = int(resources.get('GPU', 0))
    total_cpus = int(resources.get('CPU', 0))
    
    print(f"✅ Ray cluster connected:")
    print(f"   Nodes: {len(alive_nodes)}")
    print(f"   Total GPUs: {total_gpus}")
    print(f"   Total CPUs: {total_cpus}")
    
    # Print detailed node info
    for i, node in enumerate(alive_nodes):
        hostname = node.get('NodeManagerHostname', 'unknown')
        node_resources = node.get('Resources', {})
        print(f"   Node {i+1}: {hostname} - {int(node_resources.get('GPU', 0))} GPUs, {int(node_resources.get('CPU', 0))} CPUs")
    
    timer.record("cluster_info_complete", f"Cluster info gathered: {len(alive_nodes)} nodes, {total_gpus} GPUs")
    
    # Load Alpaca dataset - only first 1000 instructions for 405B
    timer.record("dataset_load_start", "Starting Alpaca dataset loading")
    print(f"\n📚 Loading Alpaca dataset...")
    
    # # Process first 500 instructions at first
    # FINAL version: to test 20000 instructions 
    
    alpaca_prompts = load_alpaca_dataset(
        min_length=5,      # Minimum 5 words
        max_length=100,    # Maximum 100 words
        num_samples=500,  # Process first 1000 instructions for 405B
        cache_dir=None
    )
    
    print(f"✅ Loaded {len(alpaca_prompts)} Alpaca instructions (from 52,002 total)")
    timer.record("dataset_load_complete", f"Alpaca dataset loaded: {len(alpaca_prompts)} instructions")
    
    # Show sample prompts
    print(f"\n📝 Sample Alpaca prompts:")
    for i, prompt in enumerate(alpaca_prompts[:3]):
        print(f"   {i+1}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    
    # Create optimized sampling params for Alpaca instructions
    timer.record("config_start", "Starting model configuration")
    sampling_params = SamplingParams(
        temperature=0.7,    # Slightly lower for more focused responses
        top_p=0.9,         # Good balance for instruction following
        max_tokens=100,    # Reasonable response length
        stop=["\n\n", "Human:", "Assistant:"]  # Stop sequences
    )
    
    # Cross-node configuration for 405B model with 16 H100 GPUs (94GB each)
    # Please test the following profiles:
    # TP=8, PP=2, instances = 1
    # TP=4, PP=4, instances = 1
    # TP=16, PP=1, instances = 1
    
    # if you have time, test TP=2, PP=8, instances = 1
    if total_gpus >= 16:
        # 405B model with TP=8, PP=2, FP8 quantization
        # This configuration balances memory efficiency and performance
        tensor_parallel_size = 8   # 8-way tensor parallelism
        pipeline_parallel_size = 2  # 2-way pipeline parallelism
        num_instances = 1          # Single instance for maximum efficiency
        gpus_per_instance = 16     # Use all 16 GPUs for one instance
        
        print(f"🎯 16-GPU 405B Configuration (TP8×PP2): 1 instance × (8 TP × 2 PP) = 16 GPUs across {len(alive_nodes)} nodes")
        print(f"   Total memory: 16×H100 (94GB) = 1,504GB total")
        print(f"   Memory per TP group: 8×94GB = 752GB per pipeline stage")
        print(f"   Pipeline stages: 2 (each stage uses 8 GPUs)")
        
    elif total_gpus >= 12:
        tensor_parallel_size = 4   
        pipeline_parallel_size = 3  
        num_instances = 1           
        gpus_per_instance = 12
        print(f"🎯 Cross-node 405B Configuration: 1 instance × (4 TP × 3 PP) = 12 GPUs across {len(alive_nodes)} nodes")
    elif total_gpus >= 8:
        tensor_parallel_size = 4
        pipeline_parallel_size = 2  
        num_instances = 1
        gpus_per_instance = 8
        print(f"🎯 Configuration: 1 instance × (4 TP × 2 PP) = 8 GPUs across nodes")
    else:
        print(f"❌ Need at least 8 GPUs for 405B model, found {total_gpus}")
        raise ValueError("Insufficient GPUs for 405B model")
    
    print(f"   tensor_parallel_size: {tensor_parallel_size}")
    print(f"   pipeline_parallel_size: {pipeline_parallel_size}")
    print(f"   num_instances: {num_instances}")
    print(f"   gpus_per_instance: {gpus_per_instance}")
    timer.record("config_complete", f"Model configuration set: TP={tensor_parallel_size}, PP={pipeline_parallel_size}")
    
    # Create LLM predictor class optimized for Alpaca instructions
    class AlpacaLLMPredictor:
        def __init__(self):
            # Record vLLM initialization start
            timer.record("vllm_init_start", f"Starting vLLM 405B initialization (TP={tensor_parallel_size}, PP={pipeline_parallel_size})")
            print(f"🏗️  Initializing 405B LLM for Alpaca instructions (TP={tensor_parallel_size}, PP={pipeline_parallel_size})")
            
            # CUDA environment fix
            import os
            print(f"   CUDA_VISIBLE_DEVICES in actor: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
            
            if not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                print("   Unset CUDA_VISIBLE_DEVICES to let vLLM auto-detect")
            
            # Optimized configuration for 405B model with TP8×PP2 + FP8 quantization
            self.llm = LLM(
                model="/mnt/REPACSS/work/LLM/Llama-3.1-405B",
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                
                # FP8 quantization for memory efficiency
                quantization="fp8",           # 🔥 Enable FP8 quantization
                
                # Optimized for 405B model on 16×H100 with quantization
                gpu_memory_utilization=0.9,  # Slightly reduced for FP8 overhead
                max_model_len=2048,           # Conservative for stability
                trust_remote_code=True,
                enforce_eager=True,
                
                # Pipeline optimized settings
                swap_space=4,                 # Reduced due to quantization savings
                max_num_batched_tokens=2048,  # Conservative for pipeline
                max_num_seqs=4,               # Reduced for single instance
                
                # Cross-node optimizations
                enable_prefix_caching=True,
                disable_custom_all_reduce=False,
                distributed_executor_backend="ray",
                
                # Additional optimizations for TP8×PP2 setup
                load_format="auto",
                dtype="auto",                 # Let vLLM choose optimal dtype with FP8
            )
            print("✅ 405B LLM initialized for Alpaca instruction following")
            timer.record("vllm_init_complete", "vLLM 405B initialization completed successfully")
            
            # Track batch processing
            self.processed_batches = 0
            self.first_token_generated = False
        
        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            batch_size = len(batch['text'])
            self.processed_batches += 1
            
            # Record inference start for first batch
            if self.processed_batches == 1:
                timer.record("inference_start", f"Starting inference on first batch ({batch_size} instructions)")
            
            print(f"\n🔄 Processing batch #{self.processed_batches} with {batch_size} Alpaca instructions")
            
            prompt = []
            generated_text = []
            processing_time = []
            
            for i, instruction in enumerate(batch["text"]):
                try:
                    global_instruction_num = (self.processed_batches - 1) * batch_size + i + 1
                    
                    print(f"\n{'='*80}")
                    print(f"📝 Processing Global Instruction #{global_instruction_num} (Batch {self.processed_batches}, Item {i+1}/{batch_size})")
                    print(f"{'='*80}")
                    print(f"Instruction: {instruction}")
                    print(f"\n⏳ Generating response...")
                    
                    # Record time for this specific instruction
                    instruction_start_time = time.time()
                    
                    # Format instruction with proper prompt template
                    formatted_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
                    
                    outputs = self.llm.generate([formatted_prompt], sampling_params)
                    
                    instruction_end_time = time.time()
                    duration = instruction_end_time - instruction_start_time
                    
                    # Record first token generation
                    if not self.first_token_generated and outputs and len(outputs) > 0:
                        output = outputs[0]
                        if hasattr(output, 'outputs') and len(output.outputs) > 0:
                            response = ' '.join([o.text for o in output.outputs])
                            if response.strip():  # Only if we actually got a response
                                timer.record("first_token_generated", f"First token generated for instruction #{global_instruction_num}")
                                self.first_token_generated = True
                    
                    for output in outputs:
                        prompt.append(instruction)  # Store original instruction
                        response = ' '.join([o.text for o in output.outputs])
                        generated_text.append(response)
                        processing_time.append(duration)
                        
                        # Real-time display of result
                        tokens = len(response.split())
                        print(f"\n✅ COMPLETED in {duration:.2f}s")
                        print(f"🤖 Response ({tokens} tokens, {tokens/duration:.2f} tokens/sec):")
                        print(f"{'─'*60}")
                        print(response)
                        print(f"{'─'*60}")
                    
                except Exception as e:
                    global_instruction_num = (self.processed_batches - 1) * batch_size + i + 1
                    print(f"\n❌ ERROR processing global instruction #{global_instruction_num}: {e}")
                    prompt.append(instruction)
                    generated_text.append("[Generation failed]")
                    processing_time.append(0.0)
            
            print(f"\n✅ Batch #{self.processed_batches} processing completed!")
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "processing_time": processing_time,
            }
    
    # Cross-node placement group strategy
    timer.record("placement_group_start", "Creating cross-node placement group")
    def scheduling_strategy_fn():
        print(f"📋 Creating cross-node placement group for {gpus_per_instance} GPUs across {len(alive_nodes)} nodes")
        
        if len(alive_nodes) >= 4 and gpus_per_instance == 16:
            # 🔥 OPTIMIZED: 16 GPUs across 4 nodes for TP8×PP2 configuration
            # Pipeline Stage 1: 8 GPUs (TP=8) across Node1+Node2 (4+4)
            # Pipeline Stage 2: 8 GPUs (TP=8) across Node3+Node4 (4+4)
            bundles = []
            for i in range(4):  # 4 nodes
                bundles.extend([{"GPU": 1, "CPU": 2}] * 4)  # 4 GPUs per node, 2 CPUs per GPU
            strategy = "SPREAD"
            print(f"   Using SPREAD strategy: 4 GPUs per node across 4 nodes")
            print(f"   Pipeline distribution:")
            print(f"     Stage 1 (TP=8): Node1(rpg-93-6) + Node2(rpg-93-8) = 8 GPUs")
            print(f"     Stage 2 (TP=8): Node3(rpg-93-5) + Node4(rpg-93-4) = 8 GPUs")
            
        elif len(alive_nodes) >= 3 and gpus_per_instance == 12:
            # Spread 12 GPUs across 3 nodes (4 GPUs per node)
            bundles = []
            for i in range(3):  # 3 nodes
                bundles.extend([{"GPU": 1, "CPU": 1}] * 4)  # 4 GPUs per node
            strategy = "SPREAD"
            print(f"   Using SPREAD strategy: 4 GPUs per node across 3 nodes")
            
        elif len(alive_nodes) >= 2 and gpus_per_instance == 8:
            # Spread 8 GPUs across 2 nodes (4 GPUs per node)
            bundles = []
            for i in range(2):  # 2 nodes
                bundles.extend([{"GPU": 1, "CPU": 1}] * 4)  # 4 GPUs per node
            strategy = "SPREAD"
            print(f"   Using SPREAD strategy: 4 GPUs per node across 2 nodes")
            
        else:
            bundles = [{"GPU": 1, "CPU": 1}] * gpus_per_instance
            strategy = "PACK"
            print(f"   Using PACK strategy: {gpus_per_instance} GPUs")
        
        pg = ray.util.placement_group(bundles, strategy=strategy)
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))
    
    # Configure resources
    resources_kwarg = {
        "num_gpus": 0,
        "ray_remote_args_fn": scheduling_strategy_fn
    }
    print(f"📌 Using cross-node placement group for {gpus_per_instance} GPUs")
    timer.record("placement_group_complete", "Cross-node placement group created")
    
    # Create Ray dataset from Alpaca prompts
    timer.record("dataset_create_start", "Creating Ray dataset from prompts")
    print(f"\n📊 Creating Ray dataset from {len(alpaca_prompts)} Alpaca instructions...")
    ds = ray.data.from_items([{"text": prompt} for prompt in alpaca_prompts])
    print(f"✅ Dataset created with {ds.count()} items")
    timer.record("dataset_create_complete", f"Ray dataset created with {ds.count()} items")
    
    # Start distributed inference
    timer.record("map_batches_start", "Starting distributed inference with map_batches")
    print(f"\n🚀 Starting 405B distributed inference on Alpaca dataset...")
    print(f"   Configuration: TP={tensor_parallel_size} × PP={pipeline_parallel_size} with FP8 quantization")
    print(f"   Concurrency: {num_instances} instance")
    print(f"   Batch size: 4")  # Conservative for TP8×PP2 + FP8
    print(f"   Total GPUs: {gpus_per_instance}")
    print(f"   Nodes: {len(alive_nodes)}")
    print(f"   Total instructions: {len(alpaca_prompts)} (first 1000 from Alpaca)")
    print(f"   Memory optimization: FP8 quantization reduces memory by ~50%")
    print(f"   Pipeline stages: 2 stages × 8 GPUs each = 16 GPUs total")
    
    start_time = time.time()
    
    # batch_size = 256 非常保守
    # You can try batch_size = 1028 if you have time
    
    
    ds = ds.map_batches(
        AlpacaLLMPredictor,
        concurrency=num_instances,
        batch_size = 256,  # Conservative batch size for TP8×PP2 stability
        **resources_kwarg,
    )
    
    # Execute and collect results
    print("⏳ Executing Alpaca instruction inference with real-time output...")
    print(f"📺 Watch below for live instruction processing...")
    outputs = ds.take_all()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    timer.record("inference_complete", f"All inference completed: {len(outputs)} results in {total_time:.2f}s")
    
    print(f"✅ Alpaca inference completed in {total_time:.2f} seconds!")
    
    # Process and compile final results summary
    timer.record("results_processing_start", "Processing and compiling final results")
    print(f"\n📊 Final Results Summary:")
    print("=" * 100)
    
    all_results = []
    total_tokens = 0
    total_processing_time = 0
    
    for i, output in enumerate(outputs):
        # Handle both single values and lists for each field
        instruction = output["prompt"]
        response = output["generated_text"]
        
        # Fix: Handle processing_time as either float or list
        proc_time_raw = output.get("processing_time", 0.0)
        if isinstance(proc_time_raw, list) and len(proc_time_raw) > 0:
            proc_time = proc_time_raw[0]
        elif isinstance(proc_time_raw, (int, float)):
            proc_time = proc_time_raw
        else:
            proc_time = 0.0
        
        # Handle response as either string or list
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
        elif not isinstance(response, str):
            response = str(response)
        
        # Handle instruction as either string or list  
        if isinstance(instruction, list) and len(instruction) > 0:
            instruction = instruction[0]
        elif not isinstance(instruction, str):
            instruction = str(instruction)
        
        tokens = len(response.split())
        total_tokens += tokens
        total_processing_time += proc_time
        
        # Create result record
        result_record = {
            "instruction_id": i + 1,
            "instruction": instruction,
            "response": response,
            "tokens_generated": tokens,
            "processing_time_seconds": proc_time,
            "tokens_per_second": tokens / proc_time if proc_time > 0 else 0
        }
        all_results.append(result_record)
    
    print(f"✅ All {len(all_results)} instructions processed and compiled!")
    timer.record("results_processing_complete", f"Results processing completed: {len(all_results)} results")
    
    # Calculate performance metrics with safe division
    avg_processing_time = total_processing_time / len(outputs) if len(outputs) > 0 else 0
    overall_throughput = total_tokens / total_time if total_time > 0 else 0
    
    performance_metrics = {
        "total_instructions": len(outputs),
        "total_time_seconds": total_time,
        "total_tokens_generated": total_tokens,
        "average_processing_time_per_instruction": avg_processing_time,
        "overall_throughput_tokens_per_second": overall_throughput,
        "gpus_used": gpus_per_instance,
        "nodes_used": len(alive_nodes),
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "throughput_per_gpu": overall_throughput / gpus_per_instance if gpus_per_instance > 0 else 0,
        "success_rate": len(all_results) / len(alpaca_prompts) if len(alpaca_prompts) > 0 else 0
    }
    
    print(f"\n⚡ Performance Summary:")
    print(f"   Instructions processed: {performance_metrics['total_instructions']}")
    print(f"   Success rate: {performance_metrics['success_rate']:.1%}")
    print(f"   Total time: {performance_metrics['total_time_seconds']:.2f}s ({performance_metrics['total_time_seconds']/60:.1f} minutes)")
    print(f"   Total tokens: {performance_metrics['total_tokens_generated']}")
    print(f"   Overall throughput: {performance_metrics['overall_throughput_tokens_per_second']:.2f} tokens/sec")
    print(f"   Average time per instruction: {performance_metrics['average_processing_time_per_instruction']:.2f}s")
    print(f"   GPUs used: {performance_metrics['gpus_used']}")
    print(f"   Throughput per GPU: {performance_metrics['throughput_per_gpu']:.2f} tokens/sec/GPU")
    print(f"   Architecture: TP={tensor_parallel_size} × PP={pipeline_parallel_size} across {len(alive_nodes)} nodes")
    
    # Save results to JSON
    timer.record("save_results_start", "Saving results to JSON file")
    model_name = "Llama-3.1-405B"
    output_file = save_results_to_json(all_results, model_name, total_time, performance_metrics, timer.timestamps)
    timer.record("save_results_complete", f"Results saved to {output_file}")
    
    print(f"\n🎉 SUCCESS: 405B model successfully processed {len(alpaca_prompts)} Alpaca instructions (first 1000)!")
    print(f"   Cross-node distributed inference working perfectly!")
    print(f"   Results saved to: {output_file}")
    print(f"   Estimated total time for all 52K instructions: {(total_time * 52002 / len(alpaca_prompts) / 3600):.1f} hours")

except Exception as e:
    timer.record("error_occurred", f"Error encountered: {str(e)}")
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Troubleshooting suggestions
    if "placement" in str(e).lower() or "resource" in str(e).lower():
        print(f"\n💡 Placement group troubleshooting:")
        print(f"1. Check if all worker nodes are properly connected with 'ray status'")
        print(f"2. Ensure each node has exactly 4 GPUs available")
        print(f"3. Verify all SLURM jobs are running with 'squeue -u $USER'")
    
    if "OutOfMemoryError" in str(e) or "CUDA out of memory" in str(e):
        print(f"\n💡 Memory optimization suggestions for TP8×PP2 + FP8 setup:")
        print(f"1. Reduce gpu_memory_utilization from 0.85 to 0.80")
        print(f"2. Reduce max_model_len from 1024 to 512")
        print(f"3. Reduce batch_size from 4 to 2 or 1")
        print(f"4. Try different quantization: 'awq' or 'gptq' instead of 'fp8'")
        print(f"5. Reduce max_num_seqs from 4 to 2")
        print(f"6. If still failing, try TP=16, PP=1 configuration")

finally:
    timer.record("script_end", "Python script execution completed")
    
    # Print comprehensive timing summary
    timer.print_summary()
    
    try:
        ray.shutdown()
        timer.record("ray_shutdown", "Ray connection closed")
        print("🔌 Ray connection closed")
    except:
        pass