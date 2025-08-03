"""
vLLM Distributed Engine - Distributed inference engine
Supports distributed inference with various configurations
"""

import os
import time
import ray
import numpy as np
from typing import Dict, List, Any, Optional
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

class VLLMDistributedEngine:
    """vLLM distributed inference engine"""
    
    def __init__(self, config: Dict[str, Any], timer):
        self.config = config
        self.timer = timer
        self.model_path = config['model_path']
        self.tensor_parallel_size = config['tensor_parallel_size']
        self.pipeline_parallel_size = config['pipeline_parallel_size']
        self.concurrency = config['concurrency']
        self.batch_size = config['batch_size']
        self.verbose = config.get('verbose', False)
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9),
            max_tokens=config.get('max_tokens', 100),
            stop=["\n\n", "Human:", "Assistant:", "###"]
        )
        
        self.llm_predictor = None
        
    def run_benchmark(self, prompts: List[str]) -> Optional[Dict[str, Any]]:
        """Run benchmark"""
        try:
            # Initialize Ray cluster
            if not self._initialize_ray():
                return None
            
            # Create Ray dataset
            dataset = self._create_dataset(prompts)
            if dataset is None:
                return None
            
            # Run inference
            return self._run_inference(dataset, prompts)
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _initialize_ray(self) -> bool:
        """Initialize Ray cluster"""
        try:
            self.timer.record("ray_init_start", "Starting Ray cluster connection")
            
            # Check Ray version
            print(f"📦 Ray version: {ray.__version__}")
            if Version(ray.__version__) < Version("2.22.0"):
                print("❌ Ray version must be at least 2.22.0")
                return False
            
            # Check if Ray is already initialized
            if not ray.is_initialized():
                print("🔄 Ray not initialized, connecting to cluster...")
                # Connect to Ray cluster
                ray.init(address="auto", ignore_reinit_error=True)
            else:
                print("✅ Ray already initialized")
            
            # Wait for cluster to stabilize
            import time
            time.sleep(2)
            
            # Get cluster information - multiple attempts
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    nodes = ray.nodes()
                    cluster_resources = ray.cluster_resources()
                    available_resources = ray.available_resources()
                    alive_nodes = [n for n in nodes if n['Alive']]
                    
                    # Try multiple ways to get GPU count
                    cluster_gpus = int(cluster_resources.get('GPU', 0))
                    available_gpus = int(available_resources.get('GPU', 0))
                    
                    # Calculate total GPUs from node information
                    node_gpus = 0
                    for node in alive_nodes:
                        node_resources = node.get('Resources', {})
                        node_gpus += int(node_resources.get('GPU', 0))
                    
                    total_gpus = max(cluster_gpus, available_gpus, node_gpus)
                    total_cpus = int(cluster_resources.get('CPU', 0))
                    
                    print(f"🔍 Ray Cluster Info (Attempt {attempt + 1}):")
                    print(f"   Cluster GPUs: {cluster_gpus}")
                    print(f"   Available GPUs: {available_gpus}")
                    print(f"   Node GPUs: {node_gpus}")
                    print(f"   Final GPU count: {total_gpus}")
                    print(f"   Nodes: {len(alive_nodes)}")
                    print(f"   Total CPUs: {total_cpus}")
                    
                    if total_gpus > 0:
                        break
                    elif attempt < max_retries - 1:
                        print(f"⚠️ No GPUs detected, retrying in 3 seconds...")
                        time.sleep(3)
                    
                except Exception as e:
                    print(f"⚠️ Error getting cluster info (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    continue
            
            # If still no GPUs, try manual detection
            if total_gpus == 0:
                print("🔍 Manual GPU detection...")
                try:
                    # Check detailed information for each node
                    for i, node in enumerate(alive_nodes):
                        hostname = node.get('NodeManagerHostname', 'unknown')
                        node_resources = node.get('Resources', {})
                        node_gpu_count = int(node_resources.get('GPU', 0))
                        print(f"   Node {i+1} ({hostname}): {node_gpu_count} GPUs")
                        total_gpus += node_gpu_count
                    
                    print(f"   Manual count total: {total_gpus} GPUs")
                except Exception as e:
                    print(f"⚠️ Manual GPU detection failed: {e}")
            
            print(f"✅ Ray cluster connected:")
            print(f"   Nodes: {len(alive_nodes)}")
            print(f"   Total GPUs: {total_gpus}")
            print(f"   Total CPUs: {total_cpus}")
            
            # Print detailed node information
            for i, node in enumerate(alive_nodes):
                hostname = node.get('NodeManagerHostname', 'unknown')
                node_resources = node.get('Resources', {})
                print(f"   Node {i+1}: {hostname} - {int(node_resources.get('GPU', 0))} GPUs, {int(node_resources.get('CPU', 0))} CPUs")
            
            # Validate resource requirements
            required_gpus = self.tensor_parallel_size * self.pipeline_parallel_size * self.concurrency
            if total_gpus < required_gpus:
                print(f"❌ Insufficient GPUs: need {required_gpus}, have {total_gpus}")
                print(f"   Current config: TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, C={self.concurrency}")
                print(f"   Suggestion: Reduce TP/PP/Concurrency or increase cluster size")
                return False
            
            self.timer.record("ray_init_complete", f"Ray initialized: {len(alive_nodes)} nodes, {total_gpus} GPUs")
            return True
            
        except Exception as e:
            print(f"❌ Ray initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_dataset(self, prompts: List[str]):
        """Create Ray dataset"""
        try:
            self.timer.record("dataset_create_start", "Creating Ray dataset")
            
            print(f"📊 Creating Ray dataset from {len(prompts)} prompts...")
            dataset = ray.data.from_items([{"text": prompt} for prompt in prompts])
            
            self.timer.record("dataset_create_complete", f"Dataset created: {dataset.count()} items")
            return dataset
            
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            return None
    
    def _run_inference(self, dataset, prompts: List[str]) -> Dict[str, Any]:
        """Run distributed inference"""
        self.timer.record("inference_start", "Starting distributed inference")
        
        print(f"🚀 Starting distributed inference...")
        print(f"   Model: {os.path.basename(self.model_path)}")
        print(f"   TP: {self.tensor_parallel_size}, PP: {self.pipeline_parallel_size}")
        print(f"   Concurrency: {self.concurrency}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Total prompts: {len(prompts)}")
        
        start_time = time.time()
        
        try:
            # Configure resources and scheduling strategy
            resources_kwarg = self._get_resource_config()
            
            # Run map_batches
            dataset = dataset.map_batches(
                VLLMPredictor,
                fn_constructor_kwargs={
                    "model_path": self.model_path,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "pipeline_parallel_size": self.pipeline_parallel_size,
                    "sampling_params": self.sampling_params,
                    "timer": self.timer,
                    "verbose": self.verbose
                },
                concurrency=self.concurrency,
                batch_size=self.batch_size,
                **resources_kwarg
            )
            
            # Collect results
            outputs = dataset.take_all()
            end_time = time.time()
            
            self.timer.record("inference_complete", f"Inference completed: {len(outputs)} results")
            
            # Process results
            return self._process_results(outputs, start_time, end_time, prompts)
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration"""
        def scheduling_strategy_fn():
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n['Alive']]
            total_gpus = self.tensor_parallel_size * self.pipeline_parallel_size
            
            print(f"📋 Creating placement group for {total_gpus} GPUs across {len(alive_nodes)} nodes")
            
            # Choose strategy based on GPU count
            if len(alive_nodes) >= 2 and total_gpus >= 8:
                # Multi-node distribution strategy
                gpus_per_node = 4  # Assume 4 GPUs per node
                bundles = []
                for i in range(len(alive_nodes)):
                    if total_gpus > 0:
                        node_gpus = min(gpus_per_node, total_gpus)
                        bundles.extend([{"GPU": 1, "CPU": 2}] * node_gpus)
                        total_gpus -= node_gpus
                strategy = "SPREAD"
                print(f"   Using SPREAD strategy across {len(alive_nodes)} nodes")
            else:
                # Single node strategy
                bundles = [{"GPU": 1, "CPU": 1}] * total_gpus
                strategy = "PACK"
                print(f"   Using PACK strategy on single node")
            
            pg = ray.util.placement_group(bundles, strategy=strategy)
            return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True))
        
        return {
            "num_gpus": 0,
            "ray_remote_args_fn": scheduling_strategy_fn
        }
    
    def _process_results(self, outputs: List[Dict], start_time: float, 
                        end_time: float, prompts: List[str]) -> Dict[str, Any]:
        """Process inference results"""
        self.timer.record("results_processing_start", "Processing results")
        
        total_time = end_time - start_time
        all_results = []
        total_tokens = 0
        total_processing_time = 0
        
        for i, output in enumerate(outputs):
            # Process output format
            instruction = self._extract_field(output, "prompt", prompts[i] if i < len(prompts) else "")
            response = self._extract_field(output, "generated_text", "[No response]")
            proc_time = self._extract_field(output, "processing_time", 0.0)
            
            if isinstance(proc_time, list) and len(proc_time) > 0:
                proc_time = proc_time[0]
            elif not isinstance(proc_time, (int, float)):
                proc_time = 0.0
                
            tokens = len(response.split()) if isinstance(response, str) else 0
            total_tokens += tokens
            total_processing_time += proc_time
            
            result_record = {
                "instruction_id": i + 1,
                "instruction": instruction,
                "response": response,
                "tokens_generated": tokens,
                "processing_time_seconds": proc_time,
                "tokens_per_second": tokens / proc_time if proc_time > 0 else 0
            }
            all_results.append(result_record)
        
        # Calculate performance metrics
        performance_metrics = {
            "total_instructions": len(outputs),
            "total_time_seconds": total_time,
            "total_tokens_generated": total_tokens,
            "average_processing_time_per_instruction": total_processing_time / len(outputs) if len(outputs) > 0 else 0,
            "overall_throughput_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "success_rate": len(all_results) / len(prompts) if len(prompts) > 0 else 0
        }
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Instructions: {performance_metrics['total_instructions']}")
        print(f"   Total time: {performance_metrics['total_time_seconds']:.2f}s")
        print(f"   Throughput: {performance_metrics['overall_throughput_tokens_per_second']:.2f} tokens/sec")
        print(f"   Success rate: {performance_metrics['success_rate']:.1%}")
        
        self.timer.record("results_processing_complete", f"Results processed: {len(all_results)} items")
        
        return {
            "results": all_results,
            "performance_metrics": performance_metrics,
            "configuration": {
                "model_path": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "concurrency": self.concurrency,
                "batch_size": self.batch_size
            }
        }
    
    def _extract_field(self, output: Dict, field: str, default: Any) -> Any:
        """Extract field from output"""
        value = output.get(field, default)
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value


class VLLMPredictor:
    """vLLM predictor class"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int, 
                 pipeline_parallel_size: int, sampling_params: SamplingParams,
                 timer, verbose: bool = False):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.sampling_params = sampling_params
        self.timer = timer
        self.verbose = verbose
        
        self.processed_batches = 0
        self.first_token_generated = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model"""
        try:
            self.timer.record("vllm_init_start", f"Initializing vLLM model (TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size})")
            
            print(f"🏗️ Initializing vLLM model...")
            print(f"   Model: {self.model_path}")
            print(f"   TP: {self.tensor_parallel_size}, PP: {self.pipeline_parallel_size}")
            
            # Clean up CUDA environment variables
            if not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Choose configuration based on model size
            model_name = os.path.basename(self.model_path).lower()
            
            if "405b" in model_name:
                # 405B model configuration
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    quantization="fp8",
                    gpu_memory_utilization=0.85,
                    max_model_len=2048,
                    trust_remote_code=True,
                    enforce_eager=True,
                    swap_space=4,
                    max_num_batched_tokens=1024,
                    max_num_seqs=4,
                    enable_prefix_caching=True,
                    distributed_executor_backend="ray",
                    load_format="auto",
                    dtype="auto"
                )
            elif "70b" in model_name:
                # 70B model configuration
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    gpu_memory_utilization=0.90,
                    max_model_len=4096,
                    trust_remote_code=True,
                    enforce_eager=True,
                    max_num_batched_tokens=2048,
                    max_num_seqs=8,
                    enable_prefix_caching=True,
                    distributed_executor_backend="ray"
                )
            elif "deepseek" in model_name or "ds-r1" in model_name:
                # DeepSeek V2 (ds-r1) model configuration
                print(f"   Applying DeepSeek V2 model configuration...")
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    quantization="fp8",
                    gpu_memory_utilization=0.90,
                    max_model_len=4096,
                    trust_remote_code=True,
                    enforce_eager=True,
                    max_num_batched_tokens=2048,
                    max_num_seqs=8,
                    enable_prefix_caching=True,
                    distributed_executor_backend="ray"
                )
            else:
                # Small model configuration (8B, 3B, etc.)
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    pipeline_parallel_size=self.pipeline_parallel_size,
                    gpu_memory_utilization=0.95,
                    max_model_len=8192,
                    trust_remote_code=True,
                    max_num_batched_tokens=4096,
                    max_num_seqs=16,
                    enable_prefix_caching=True,
                    distributed_executor_backend="ray"
                )
            
            print("✅ vLLM model initialized successfully")
            self.timer.record("vllm_init_complete", "vLLM model initialization completed")
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            raise
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        """Process batch data"""
        batch_size = len(batch['text'])
        self.processed_batches += 1
        
        if self.processed_batches == 1:
            self.timer.record("inference_start", f"Starting inference on first batch ({batch_size} prompts)")
        
        if self.verbose:
            print(f"\n🔄 Processing batch #{self.processed_batches} with {batch_size} prompts")
        
        prompts = []
        generated_texts = []
        processing_times = []
        
        for i, instruction in enumerate(batch["text"]):
            try:
                global_prompt_num = (self.processed_batches - 1) * batch_size + i + 1
                
                if self.verbose:
                    print(f"📝 Processing prompt #{global_prompt_num}")
                
                instruction_start_time = time.time()
                
                # Format prompt
                formatted_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
                
                # Generate response
                outputs = self.llm.generate([formatted_prompt], self.sampling_params)
                
                instruction_end_time = time.time()
                duration = instruction_end_time - instruction_start_time
                
                # Record first token generation
                if not self.first_token_generated and outputs and len(outputs) > 0:
                    output = outputs[0]
                    if hasattr(output, 'outputs') and len(output.outputs) > 0:
                        response = ' '.join([o.text for o in output.outputs])
                        if response.strip():
                            self.timer.record("first_token_generated", f"First token generated for prompt #{global_prompt_num}")
                            self.first_token_generated = True
                
                # Process output
                for output in outputs:
                    prompts.append(instruction)
                    response = ' '.join([o.text for o in output.outputs])
                    generated_texts.append(response)
                    processing_times.append(duration)
                    
                    if self.verbose:
                        tokens = len(response.split())
                        print(f"✅ Completed in {duration:.2f}s ({tokens} tokens, {tokens/duration:.2f} tokens/sec)")
                        
            except Exception as e:
                print(f"❌ Error processing prompt #{global_prompt_num}: {e}")
                prompts.append(instruction)
                generated_texts.append("[Generation failed]")
                processing_times.append(0.0)
        
        if self.verbose:
            print(f"✅ Batch #{self.processed_batches} completed!")
        
        return {
            "prompt": prompts,
            "generated_text": generated_texts,
            "processing_time": processing_times,
        }