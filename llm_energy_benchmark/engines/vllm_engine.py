"""
vLLM inference engine for benchmarking energy consumption.
"""

import time
import torch
import os
import pynvml

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Please install it using: pip install vllm")
    print("If you are using a CPU-only machine, you may need to use: pip install vllm-python")

class VLLMEngine:
    """vLLM inference engine wrapper for benchmarking."""
    
    def __init__(self):
        """Initialize the vLLM engine."""
        self.available = VLLM_AVAILABLE
    
    def setup_model(self, model_path, gpu_memory_utilization=0.9, max_model_len=None):
        """Initialize a vLLM model."""
        if not self.available:
            print("vLLM is not available. Please install it first.")
            return None
            
        try:
            print(f"Loading model from {model_path} with vLLM...")
            
            # Clear GPU cache first
            torch.cuda.empty_cache()
            
            # Get available GPU count
            gpu_count = torch.cuda.device_count()
            
            # # Use integer instead of "auto" for tensor_parallel_size
            # if gpu_count > 1:
            #     tensor_parallel_size = gpu_count  # Use all available GPUs
            # else:
            #     tensor_parallel_size = 1  # Only 1 GPU
            tensor_parallel_size = 1
                
            print(f"Setting tensor_parallel_size to {tensor_parallel_size} (available GPUs: {gpu_count})")
            
            # 设置环境变量允许覆盖最大长度限制
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            
            # 如果没有指定max_model_len，从模型配置中读取
            if max_model_len is None:
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_path)
                    if hasattr(config, 'max_position_embeddings'):
                        max_model_len = config.max_position_embeddings
                    else:
                        max_model_len = 1024  # 默认值
                except Exception as e:
                    print(f"Warning: Could not read max_position_embeddings from config, using default: {e}")
                    max_model_len = 1024
            
            print(f"Using max_model_len: {max_model_len}")
            
            # Initialize model with specified parameters
            from vllm import LLM
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True
            )
            print(f"Model loaded successfully from {model_path}")
            
            self.llm = llm

            # 使用 pynvml 获取加载后显存
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.load_memory_mb = round(mem_info.used / 1024 / 1024, 2)
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"Warning: Failed to get memory info via pynvml: {e}")
                self.load_memory_mb = 0.0

            print(f"GPU memory after loading model: {self.load_memory_mb} MB")

            return llm
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_inference(self, prompts, batch_size, max_tokens=200, temperature=0.7):
        """Run batched inference with vLLM."""
        if not hasattr(self, 'llm'):
            return []
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Process prompts in batches of batch_size
        all_results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate with vLLM
            outputs = self.llm.generate(batch_prompts, sampling_params)
            all_results.extend(outputs)
        
        return all_results
    
    def run_benchmark(self, prompts, num_samples, batch_size, max_tokens):
        """Run multiple inference passes for proper benchmarking and collect memory info."""
        if not hasattr(self, 'llm'):
            return [], 0, 0, {}

        all_outputs = []
        start_time = time.time()

        # Calculate total prompts needed
        total_needed = num_samples

        # Prepare enough prompts (repeating if necessary)
        full_prompts = []
        while len(full_prompts) < total_needed:
            full_prompts.extend(prompts)
        full_prompts = full_prompts[:total_needed]

        # Run inference in batches
        for i in range(num_samples):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_prompts = full_prompts[start_idx:end_idx]
            outputs = self.run_inference(batch_prompts, batch_size, max_tokens)
            all_outputs.extend(outputs)

        end_time = time.time()

        # 使用 pynvml 获取 GPU 显存数据
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 通常是第0块GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_after_load = round(mem_info.used / 1024 / 1024, 2)
            gpu_memory_total = round(mem_info.total / 1024 / 1024, 2)
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"Warning: Failed to get memory info via pynvml: {e}")
            gpu_memory_after_load = 0.0
            gpu_memory_total = 0.0

        memory_info = {
            "gpu_memory_after_load_mb": gpu_memory_after_load,
            "gpu_memory_total_mb": gpu_memory_total
        }

        return all_outputs, start_time, end_time, memory_info


    
    def estimate_tokens(self, outputs):
        """Estimate token count from vLLM outputs."""
        tokens_so_far = 0
        for output in outputs:
            # Safely get output text
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                text = output.outputs[0].text
                tokens_so_far += int(len(text.split()) * 1.3)  # Rough approximation
        return tokens_so_far
    
    def print_setup_instructions(self):
        """Print instructions for setting up models for vLLM."""
        print("\nModel Setup Instructions for vLLM:")
        print("="*50)
        print("vLLM supports many model types including HuggingFace models.")
        print("Here's how to prepare the models you mentioned:")
        
        print("\n1. TinyLlama:")
        print("   - HuggingFace ID: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        print("   - Download: No action needed, vLLM will download automatically")
        
        print("\n2. Llama 2:")
        print("   - HuggingFace ID: 'meta-llama/Llama-2-7b-chat-hf'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        print("   - Download: Can pre-download with 'huggingface-cli download meta-llama/Llama-2-7b-chat-hf'")
        
        print("\n3. Llama 3.1:")
        print("   - HuggingFace ID: 'meta-llama/Meta-Llama-3.1-8B'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        print("   - Download: Can pre-download with 'huggingface-cli download meta-llama/Meta-Llama-3.1-8B'")
        
        print("\nIf you need to login to HuggingFace first:")
        print("huggingface-cli login")
        
        print("\nInstall vLLM with: pip install vllm")
        print("="*50)