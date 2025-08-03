"""
Dataset Loader - Multi-dataset loader supporting various datasets
Supports Alpaca, Dolly 15K, LongBench, HumanEval datasets
"""

from typing import List, Optional, Dict, Any
import random
import json

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

class DatasetLoader:
    """Dataset loader class"""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        
    def load_dataset(self, dataset_name: str, num_samples: int = 1000, 
                    min_length: int = 5, max_length: int = 100) -> List[str]:
        """
        Load specified dataset
        
        Args:
            dataset_name: Dataset name (alpaca, dolly, longbench, humaneval)
            num_samples: Number of samples
            min_length: Minimum word count
            max_length: Maximum word count
            
        Returns:
            List[str]: List of prompts
        """
        dataset_name = dataset_name.lower().strip()
        
        if dataset_name == "alpaca":
            return self._load_alpaca(num_samples, min_length, max_length)
        elif dataset_name == "dolly":
            return self._load_dolly(num_samples, min_length, max_length)
        elif dataset_name == "longbench":
            return self._load_longbench(num_samples, min_length, max_length)
        elif dataset_name == "humaneval":
            return self._load_humaneval(num_samples, min_length, max_length)
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            return self._get_fallback_prompts()
    
    def _load_alpaca(self, num_samples: int, min_length: int, max_length: int) -> List[str]:
        """Load Alpaca dataset"""
        if not DATASETS_AVAILABLE:
            return self._get_fallback_prompts()
        
        try:
            print("🔄 Loading Alpaca dataset...")
            dataset = load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir)
            
            prompts = []
            if "train" in dataset:
                for item in dataset["train"]:
                    if "instruction" in item and item["instruction"].strip():
                        instruction = item["instruction"].strip()
                        
                        # Add input context (if available)
                        if "input" in item and item["input"].strip():
                            instruction = f"{instruction}\n\nContext: {item['input'].strip()}"
                        
                        prompts.append(instruction)
            
            return self._filter_and_sample(prompts, num_samples, min_length, max_length, "Alpaca")
            
        except Exception as e:
            print(f"❌ Error loading Alpaca dataset: {e}")
            return self._get_fallback_prompts()
    
    def _load_dolly(self, num_samples: int, min_length: int, max_length: int) -> List[str]:
        """Load Dolly 15K dataset"""
        if not DATASETS_AVAILABLE:
            return self._get_fallback_prompts()
        
        try:
            print("🔄 Loading Dolly 15K dataset...")
            dataset = load_dataset("databricks/databricks-dolly-15k", cache_dir=self.cache_dir)
            
            prompts = []
            if "train" in dataset:
                for item in dataset["train"]:
                    if "instruction" in item and item["instruction"].strip():
                        instruction = item["instruction"].strip()
                        
                        # Add context (if available)
                        if "context" in item and item["context"].strip():
                            instruction = f"{instruction}\n\nContext: {item['context'].strip()}"
                        
                        prompts.append(instruction)
            
            return self._filter_and_sample(prompts, num_samples, min_length, max_length, "Dolly 15K")
            
        except Exception as e:
            print(f"❌ Error loading Dolly dataset: {e}")
            return self._get_fallback_prompts()
    
    def _load_longbench(self, num_samples: int, min_length: int, max_length: int) -> List[str]:
        """Load LongBench dataset"""
        if not DATASETS_AVAILABLE:
            return self._get_longbench_fallback()
        
        try:
            print("🔄 Loading LongBench dataset...")
            # LongBench has multiple subtasks, we select several main ones
            subtasks = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]
            
            all_prompts = []
            for subtask in subtasks:
                try:
                    dataset = load_dataset("THUDM/LongBench", subtask, cache_dir=self.cache_dir)
                    if "test" in dataset:
                        for item in dataset["test"]:
                            if "input" in item:
                                all_prompts.append(item["input"])
                except Exception as e:
                    print(f"⚠️ Error loading LongBench subtask {subtask}: {e}")
                    continue
            
            if not all_prompts:
                return self._get_longbench_fallback()
            
            return self._filter_and_sample(all_prompts, num_samples, min_length, max_length, "LongBench")
            
        except Exception as e:
            print(f"❌ Error loading LongBench dataset: {e}")
            return self._get_longbench_fallback()
    
    def _load_humaneval(self, num_samples: int, min_length: int, max_length: int) -> List[str]:
        """Load HumanEval dataset"""
        if not DATASETS_AVAILABLE:
            return self._get_humaneval_fallback()
        
        try:
            print("🔄 Loading HumanEval dataset...")
            dataset = load_dataset("openai/openai_humaneval", cache_dir=self.cache_dir)
            
            prompts = []
            if "test" in dataset:
                for item in dataset["test"]:
                    if "prompt" in item and item["prompt"].strip():
                        # Add instruction for programming tasks
                        coding_prompt = f"Complete the following Python function:\n\n{item['prompt']}"
                        prompts.append(coding_prompt)
            
            return self._filter_and_sample(prompts, num_samples, min_length, max_length, "HumanEval")
            
        except Exception as e:
            print(f"❌ Error loading HumanEval dataset: {e}")
            return self._get_humaneval_fallback()
    
    def _filter_and_sample(self, prompts: List[str], num_samples: int, 
                          min_length: int, max_length: int, dataset_name: str) -> List[str]:
        """Filter and sample prompts"""
        print(f"📊 Loaded {len(prompts)} raw prompts from {dataset_name}")
        
        # Filter by length
        filtered_prompts = []
        for prompt in prompts:
            words = len(prompt.split())
            if min_length <= words <= max_length:
                filtered_prompts.append(prompt)
        
        print(f"📋 Filtered to {len(filtered_prompts)} prompts between {min_length} and {max_length} words")
        
        # Random sampling
        if num_samples < len(filtered_prompts):
            sampled_prompts = random.sample(filtered_prompts, num_samples)
            print(f"🎯 Randomly sampled {num_samples} prompts from {dataset_name}")
        else:
            sampled_prompts = filtered_prompts
            print(f"✅ Using all {len(sampled_prompts)} available prompts from {dataset_name}")
        
        return sampled_prompts
    
    def _get_fallback_prompts(self) -> List[str]:
        """Generic fallback prompts"""
        return [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain quantum computing in simple terms.",
            "What are the main causes of climate change?",
            "How do neural networks learn?",
            "What is the importance of data science?",
            "Describe the evolution of programming languages."
        ]
    
    def _get_longbench_fallback(self) -> List[str]:
        """LongBench fallback prompts (longer texts)"""
        return [
            """Given the following passage about climate change, analyze the main arguments and provide a comprehensive summary:
            
Climate change refers to long-term changes in global and regional climate patterns. The primary driver of modern climate change is human activity, particularly the emission of greenhouse gases such as carbon dioxide and methane. These gases trap heat in the atmosphere, leading to global warming. The effects include rising sea levels, more frequent extreme weather events, and shifts in precipitation patterns. Scientists have reached a strong consensus that immediate action is needed to reduce emissions and mitigate these effects.""",

            """Read the following story excerpt and answer the questions about character development and themes:
            
In a small village nestled between rolling hills and a winding river, there lived an old craftsman named Elena. For forty years, she had been creating beautiful wooden furniture with her own hands. Each piece told a story, carved with precision and love. But as the world changed around her, mass-produced furniture became more popular, and fewer people appreciated handmade crafts. Elena faced a difficult decision: adapt to modern methods or continue her traditional ways, knowing her business might not survive.""",

            """Analyze the following scientific research abstract and explain its implications:
            
Recent studies have shown that artificial neural networks can exhibit emergent behaviors not explicitly programmed into their architecture. Researchers observed that when trained on diverse datasets, these networks develop internal representations that mirror certain cognitive processes found in biological systems. This phenomenon, termed 'computational emergence,' suggests that intelligence might arise from the complex interactions of simple processing units rather than from explicit programming of intelligent behaviors."""
        ]
    
    def _get_humaneval_fallback(self) -> List[str]:
        """HumanEval fallback prompts (programming tasks)"""
        return [
            """Complete the following Python function:

def fibonacci(n: int) -> int:
    \"\"\"
    Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
    
    Returns:
        The nth Fibonacci number
    \"\"\"
    # Your code here""",

            """Complete the following Python function:

def is_palindrome(s: str) -> bool:
    \"\"\"
    Check if a string is a palindrome (reads the same forwards and backwards).
    
    Args:
        s: The string to check
    
    Returns:
        True if the string is a palindrome, False otherwise
    \"\"\"
    # Your code here""",

            """Complete the following Python function:

def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    \"\"\"
    Merge two sorted lists into one sorted list.
    
    Args:
        list1: First sorted list
        list2: Second sorted list
    
    Returns:
        A new sorted list containing all elements from both input lists
    \"\"\"
    # Your code here""",

            """Complete the following Python function:

def binary_search(arr: List[int], target: int) -> int:
    \"\"\"
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted array to search in
        target: Value to search for
    
    Returns:
        Index of target if found, -1 otherwise
    \"\"\"
    # Your code here""",

            """Complete the following Python function:

def quick_sort(arr: List[int]) -> List[int]:
    \"\"\"
    Sort an array using the quicksort algorithm.
    
    Args:
        arr: Array to sort
    
    Returns:
        A new sorted array
    \"\"\"
    # Your code here"""
        ]

    def get_dataset_info(self) -> Dict[str, str]:
        """Get information about supported datasets"""
        return {
            "alpaca": "Stanford Alpaca dataset - 52K instruction-following demonstrations",
            "dolly": "Databricks Dolly 15K - High-quality human-generated instruction tuning dataset",
            "longbench": "LongBench - A bilingual, multitask benchmark for long context understanding",
            "humaneval": "HumanEval - Evaluating large language models trained on code"
        }