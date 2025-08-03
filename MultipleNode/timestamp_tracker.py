"""
Timestamp Tracker - Detailed timestamp tracking and performance analysis
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

class TimestampTracker:
    """Timestamp tracker class"""
    
    def __init__(self, log_file_prefix: str = "benchmark_timestamps"):
        self.timestamps = {}
        self.stage_durations = {}
        self.log_file_prefix = log_file_prefix
        self.log_file = f"{log_file_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
    def set_log_file_name(self, model: str, dataset: str, num_prompts: int,
                          tp: int, pp: int, conc: int, batch_size: int):
        """Set detailed log file name based on configuration"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # Sanitize model name by replacing slashes
        model_name_sanitized = model.replace('/', '_')
        self.log_file = (
            f"{self.log_file_prefix}_{model_name_sanitized}_{dataset}_"
            f"{num_prompts}p_tp{tp}_pp{pp}_c{conc}_b{batch_size}_{timestamp}.json"
        )

    def record(self, stage_name: str, description: str = ""):
        """Record timestamp for specific stage"""
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
                    "end_time": list(self.timestamps.values())[-1]["readable_time"] if self.timestamps else None,
                    "total_duration": self.get_total_duration()
                }
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save timestamps: {e}")
    
    def get_total_duration(self) -> Optional[float]:
        """Get total duration"""
        if len(self.timestamps) >= 2:
            start_key = list(self.timestamps.keys())[0]
            end_key = list(self.timestamps.keys())[-1]
            return self.get_duration(start_key, end_key)
        return None
    
    def print_summary(self):
        """Print summary of all recorded timestamps"""
        print(f"\n📊 TIMING SUMMARY")
        print("=" * 80)
        
        for stage, info in self.timestamps.items():
            print(f"   {stage}: {info['readable_time']} - {info['description']}")
        
        print(f"\n⏱️ STAGE DURATIONS")
        print("-" * 60)
        for duration_name, duration in self.stage_durations.items():
            print(f"   {duration_name}: {duration:.2f}s ({duration/60:.1f}min)")
        
        total_duration = self.get_total_duration()
        if total_duration:
            print(f"\n🕐 TOTAL TIME: {total_duration:.2f}s ({total_duration/60:.1f}min)")
        
        print(f"\n💾 Detailed timestamps saved to: {self.log_file}")
    
    def get_performance_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown information"""
        breakdown = {
            "initialization_time": 0.0,
            "inference_time": 0.0,
            "processing_time": 0.0,
            "total_time": self.get_total_duration() or 0.0
        }
        
        # Calculate initialization time
        init_stages = [k for k in self.stage_durations.keys() if "init" in k.lower()]
        breakdown["initialization_time"] = sum(self.stage_durations[stage] for stage in init_stages)
        
        # Calculate inference time
        inference_stages = [k for k in self.stage_durations.keys() if "inference" in k.lower()]
        breakdown["inference_time"] = sum(self.stage_durations[stage] for stage in inference_stages)
        
        # Calculate processing time
        processing_stages = [k for k in self.stage_durations.keys() if "processing" in k.lower()]
        breakdown["processing_time"] = sum(self.stage_durations[stage] for stage in processing_stages)
        
        return breakdown
    
    def add_checkpoint(self, name: str, description: str = ""):
        """Add checkpoint (alias)"""
        self.record(name, description)
    
    def start_timer(self, name: str, description: str = ""):
        """Start timer"""
        self.record(f"{name}_start", f"Started: {description}")
    
    def end_timer(self, name: str, description: str = ""):
        """End timer"""
        self.record(f"{name}_end", f"Completed: {description}")
        
        # Calculate total time for this timer
        start_key = f"{name}_start"
        end_key = f"{name}_end"
        if start_key in self.timestamps and end_key in self.timestamps:
            duration = self.get_duration(start_key, end_key)
            print(f"⏰ Timer '{name}' duration: {duration:.2f}s ({duration/60:.1f}min)")
            return duration
        return 0.0
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """Export timestamps to CSV file"""
        import csv
        
        if filename is None:
            filename = self.log_file.replace('.json', '.csv')
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Stage', 'Timestamp', 'Readable Time', 'Description'])
                
                for stage, info in self.timestamps.items():
                    writer.writerow([
                        stage,
                        info['timestamp'],
                        info['readable_time'],
                        info['description']
                    ])
            
            print(f"📊 Timestamps exported to CSV: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Failed to export to CSV: {e}")
            return ""
    
    def get_stage_statistics(self) -> Dict[str, float]:
        """Get stage statistics"""
        if not self.stage_durations:
            return {}
        
        durations = list(self.stage_durations.values())
        
        return {
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_duration": sum(durations) / len(durations),
            "total_stages": len(durations)
        }
    
    def print_detailed_report(self):
        """Print detailed report"""
        print(f"\n📈 DETAILED PERFORMANCE REPORT")
        print("=" * 80)
        
        # Basic information
        total_duration = self.get_total_duration()
        print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f}min)" if total_duration else "N/A")
        print(f"Total Stages: {len(self.timestamps)}")
        print(f"Log File: {self.log_file}")
        
        # Performance breakdown
        breakdown = self.get_performance_breakdown()
        print(f"\n⚡ Performance Breakdown:")
        for category, duration in breakdown.items():
            percentage = (duration / total_duration * 100) if total_duration and duration else 0
            print(f"   {category.replace('_', ' ').title()}: {duration:.2f}s ({percentage:.1f}%)")
        
        # Stage statistics
        stats = self.get_stage_statistics()
        if stats:
            print(f"\n📊 Stage Statistics:")
            print(f"   Shortest stage: {stats['min_duration']:.2f}s")
            print(f"   Longest stage: {stats['max_duration']:.2f}s")
            print(f"   Average stage: {stats['avg_duration']:.2f}s")
        
        # Slowest stage
        if self.stage_durations:
            slowest = max(self.stage_durations.items(), key=lambda x: x[1])
            print(f"\n🐌 Slowest Stage: {slowest[0]} ({slowest[1]:.2f}s)")
            
        # Fastest stage
        if self.stage_durations:
            fastest = min(self.stage_durations.items(), key=lambda x: x[1])
            print(f"⚡ Fastest Stage: {fastest[0]} ({fastest[1]:.2f}s)")
        
        print("=" * 80)