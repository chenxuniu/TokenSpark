"""
Configuration Manager - Configuration validation and management
Validates the reasonableness of TP, PP, concurrency and other parameters
"""

import ray
import time
from typing import Dict, List, Tuple, Optional

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self):
        self.cluster_info = None
        self._initialize_ray_and_update_cluster_info()
    
    def _initialize_ray_and_update_cluster_info(self):
        """Initialize Ray if needed and update cluster information"""
        try:
            if not ray.is_initialized():
                print("🔄 Ray not initialized, connecting to cluster...")
                ray.init(address="auto", ignore_reinit_error=True)
                time.sleep(2)  # Wait for connection to stabilize

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    nodes = ray.nodes()
                    cluster_resources = ray.cluster_resources()
                    available_resources = ray.available_resources()
                    alive_nodes = [n for n in nodes if n['Alive']]
                    
                    cluster_gpus = int(cluster_resources.get('GPU', 0))
                    available_gpus = int(available_resources.get('GPU', 0))
                    
                    node_gpus = 0
                    for node in alive_nodes:
                        node_resources = node.get('Resources', {})
                        node_gpus += int(node_resources.get('GPU', 0))
                    
                    total_gpus = max(cluster_gpus, available_gpus, node_gpus)
                    
                    if total_gpus > 0:
                        print(f"✅ Ray cluster connected:")
                        print(f"   Nodes: {len(alive_nodes)}")
                        print(f"   Total GPUs: {total_gpus}")
                        print(f"   Total CPUs: {int(cluster_resources.get('CPU', 0))}")
                        self.cluster_info = {
                            'total_gpus': total_gpus,
                            'total_cpus': int(cluster_resources.get('CPU', 0)),
                            'num_nodes': len(alive_nodes),
                            'nodes': alive_nodes,
                            'cluster_resources': cluster_resources,
                            'available_resources': available_resources
                        }
                        return
                    
                    print(f"⚠️ No GPUs detected yet (attempt {attempt + 1}), retrying in 3 seconds...")
                    time.sleep(3)
                
                except Exception as e:
                    print(f"⚠️ Error getting cluster info (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            
            print("❌ Failed to get cluster info after multiple retries.")
            self.cluster_info = {
                'total_gpus': 0, 'total_cpus': 0, 'num_nodes': 0, 'nodes': []
            }

        except Exception as e:
            print(f"❌ Ray initialization or cluster info fetch failed: {e}")
            import traceback
            traceback.print_exc()
            self.cluster_info = {
                'total_gpus': 0, 'total_cpus': 0, 'num_nodes': 0, 'nodes': []
            }

    def _update_cluster_info(self):
        """(DEPRECATED) Update cluster information"""
        if not self.cluster_info:
            self._initialize_ray_and_update_cluster_info()
    
    def validate_config(self, tensor_parallel: int, pipeline_parallel: int, 
                       concurrency: int) -> bool:
        """
        Validate configuration effectiveness
        
        Args:
            tensor_parallel: Tensor parallelism degree
            pipeline_parallel: Pipeline parallelism degree  
            concurrency: Number of concurrent instances
            
        Returns:
            bool: Whether the configuration is valid
        """
        if not self.cluster_info or self.cluster_info['total_gpus'] == 0:
            print("🔄 Re-checking cluster info for validation...")
            self._initialize_ray_and_update_cluster_info()
        
        total_gpus_needed = tensor_parallel * pipeline_parallel * concurrency
        available_gpus = self.cluster_info['total_gpus']
        
        # Basic resource check
        if total_gpus_needed > available_gpus:
            print(f"❌ Insufficient GPUs: need {total_gpus_needed}, have {available_gpus}")
            return False
        
        # Tensor parallelism check
        if tensor_parallel > available_gpus:
            print(f"❌ Tensor parallel size ({tensor_parallel}) exceeds available GPUs ({available_gpus})")
            return False
        
        # Pipeline parallelism check
        if pipeline_parallel > self.cluster_info['num_nodes']:
            print(f"⚠️ Pipeline parallel size ({pipeline_parallel}) exceeds number of nodes ({self.cluster_info['num_nodes']})")
            print("   This may work but is not optimal for cross-node pipeline parallelism")
        
        # Recommended configuration check
        recommendations = self.get_recommended_configs()
        current_config = (tensor_parallel, pipeline_parallel, concurrency)
        
        if current_config not in [tuple(r[:3]) for r in recommendations]:
            print(f"⚠️ Configuration {current_config} is not in recommended list")
            print("   This may work but performance might not be optimal")
        
        return True
    
    def get_recommended_configs(self) -> List[Tuple[int, int, int, str]]:
        """
        Get recommended configurations
        
        Returns:
            List[Tuple[int, int, int, str]]: (TP, PP, concurrency, description)
        """
        if not self.cluster_info:
            self._initialize_ray_and_update_cluster_info()
        
        total_gpus = self.cluster_info['total_gpus']
        num_nodes = self.cluster_info['num_nodes']
        
        recommendations = []
        
        if total_gpus >= 16:
            # 16+ GPU configurations
            recommendations.extend([
                (16, 1, 1, "Single instance, TP=16 (best for large models)"),
                (8, 2, 1, "Single instance, TP=8×PP=2 (pipeline for very large models)"),
                (8, 1, 2, "Two instances, TP=8 each (parallel processing)"),
                (4, 4, 1, "Single instance, TP=4×PP=4 (deep pipeline)"),
                (4, 2, 2, "Two instances, TP=4×PP=2 each"),
                (4, 1, 4, "Four instances, TP=4 each (maximum parallelism)")
            ])
        
        if total_gpus >= 12:
            # 12+ GPU configurations
            recommendations.extend([
                (12, 1, 1, "Single instance, TP=12"),
                (6, 2, 1, "Single instance, TP=6×PP=2"),
                (4, 3, 1, "Single instance, TP=4×PP=3"),
                (6, 1, 2, "Two instances, TP=6 each"),
                (4, 1, 3, "Three instances, TP=4 each")
            ])
        
        if total_gpus >= 8:
            # 8+ GPU configurations
            recommendations.extend([
                (8, 1, 1, "Single instance, TP=8"),
                (4, 2, 1, "Single instance, TP=4×PP=2"),
                (4, 1, 2, "Two instances, TP=4 each"),
                (2, 4, 1, "Single instance, TP=2×PP=4"),
                (2, 2, 2, "Two instances, TP=2×PP=2 each"),
                (2, 1, 4, "Four instances, TP=2 each")
            ])
        
        if total_gpus >= 4:
            # 4+ GPU configurations
            recommendations.extend([
                (4, 1, 1, "Single instance, TP=4"),
                (2, 2, 1, "Single instance, TP=2×PP=2"),
                (2, 1, 2, "Two instances, TP=2 each"),
                (1, 4, 1, "Single instance, PP=4"),
                (1, 2, 2, "Two instances, PP=2 each"),
                (1, 1, 4, "Four instances, single GPU each")
            ])
        
        if total_gpus >= 2:
            # 2+ GPU configurations
            recommendations.extend([
                (2, 1, 1, "Single instance, TP=2"),
                (1, 2, 1, "Single instance, PP=2"),
                (1, 1, 2, "Two instances, single GPU each")
            ])
        
        if total_gpus >= 1:
            # Single GPU configuration
            recommendations.append((1, 1, 1, "Single instance, single GPU"))
        
        # Filter out configurations that exceed resource limits
        valid_recommendations = []
        for tp, pp, conc, desc in recommendations:
            if tp * pp * conc <= total_gpus:
                valid_recommendations.append((tp, pp, conc, desc))
        
        # Remove duplicates and sort
        seen = set()
        unique_recommendations = []
        for rec in valid_recommendations:
            key = rec[:3]  # (tp, pp, conc)
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        # Sort by GPU usage
        unique_recommendations.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
        
        return unique_recommendations
    
    def get_model_specific_configs(self, model_name: str) -> List[Tuple[int, int, int, str]]:
        """
        Get model-specific recommended configurations
        
        Args:
            model_name: Model name
            
        Returns:
            List of recommended configurations
        """
        model_name = model_name.lower()
        all_configs = self.get_recommended_configs()
        
        if "405b" in model_name:
            # 405B model needs more GPUs
            return [config for config in all_configs if config[0] * config[1] >= 8]
        elif "70b" in model_name:
            # 70B model needs medium GPUs
            return [config for config in all_configs if config[0] * config[1] >= 4]
        elif any(size in model_name for size in ["8b", "7b", "13b"]):
            # Small to medium models
            return [config for config in all_configs if config[0] * config[1] >= 1]
        elif any(size in model_name for size in ["3b", "1b"]):
            # Small models
            return [config for config in all_configs if config[0] * config[1] >= 1]
        else:
            # Unknown model, return all configurations
            return all_configs
    
    def print_cluster_info(self):
        """Print cluster information"""
        if not self.cluster_info:
            self._initialize_ray_and_update_cluster_info()
        
        print(f"\n🖥️ Cluster Information:")
        print(f"   Total GPUs: {self.cluster_info['total_gpus']}")
        print(f"   Total CPUs: {self.cluster_info['total_cpus']}")
        print(f"   Number of nodes: {self.cluster_info['num_nodes']}")
        
        for i, node in enumerate(self.cluster_info['nodes']):
            hostname = node.get('NodeManagerHostname', 'unknown')
            node_resources = node.get('Resources', {})
            gpus = int(node_resources.get('GPU', 0))
            cpus = int(node_resources.get('CPU', 0))
            print(f"   Node {i+1}: {hostname} - {gpus} GPUs, {cpus} CPUs")
    
    def print_recommended_configs(self, model_name: Optional[str] = None):
        """Print recommended configurations"""
        if model_name:
            configs = self.get_model_specific_configs(model_name)
            print(f"\n⚙️ Recommended configurations for {model_name}:")
        else:
            configs = self.get_recommended_configs()
            print(f"\n⚙️ Recommended configurations:")
        
        print("-" * 80)
        print(f"{'TP':<4} {'PP':<4} {'Conc':<6} {'GPUs':<6} {'Description'}")
        print("-" * 80)
        
        for tp, pp, conc, desc in configs:
            total_gpus = tp * pp * conc
            print(f"{tp:<4} {pp:<4} {conc:<6} {total_gpus:<6} {desc}")
    
    def suggest_batch_sizes(self, tensor_parallel: int, pipeline_parallel: int, 
                           model_name: str) -> List[int]:
        """
        Suggest batch sizes
        
        Args:
            tensor_parallel: Tensor parallelism degree
            pipeline_parallel: Pipeline parallelism degree
            model_name: Model name
            
        Returns:
            List of suggested batch sizes
        """
        model_name = model_name.lower()
        total_gpus = tensor_parallel * pipeline_parallel
        
        if "405b" in model_name:
            # 405B model - conservative batch sizes
            if total_gpus >= 16:
                return [64, 128, 256]
            elif total_gpus >= 8:
                return [32, 64, 128]
            else:
                return [16, 32, 64]
        elif "70b" in model_name:
            # 70B model - medium batch sizes
            if total_gpus >= 8:
                return [128, 256, 512]
            elif total_gpus >= 4:
                return [64, 128, 256]
            else:
                return [32, 64, 128]
        else:
            # Small models - larger batch sizes
            if total_gpus >= 4:
                return [256, 512, 1024]
            elif total_gpus >= 2:
                return [128, 256, 512]
            else:
                return [64, 128, 256]