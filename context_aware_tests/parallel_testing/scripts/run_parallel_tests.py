"""
Parallel Test Runner
Launches multiple optimization tests simultaneously with isolated outputs
"""

import subprocess
import os
import sys
import time
import json
from typing import List, Dict
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import generate_run_id

class ParallelTestRunner:
    """Manages parallel execution of multiple test configurations"""
    
    def __init__(self, config_file: str = None):
        """Initialize the parallel test runner"""
        self.processes = []
        self.results = {}
        
        # Load configurations
        if config_file is None:
            config_file = Path(__file__).parent.parent / "configs" / "test_configurations.json"
        
        self.config_file = config_file
        self.configs = self.load_configurations()
    
    def load_configurations(self) -> Dict:
        """Load test configurations from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            return data.get('configurations', {})
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            return {}
    
    def list_configurations(self) -> None:
        """List all available configurations"""
        print("📋 AVAILABLE TEST CONFIGURATIONS:")
        print("="*60)
        
        for config_name, config in self.configs.items():
            print(f"🔧 {config_name}:")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Combinations: {config['estimated_combinations']}")
            print(f"   Est. Duration: {config['estimated_duration_minutes']} minutes")
            print(f"   Renko: {'Yes' if config['use_renko'] else 'No'}")
            print()
    
    def run_single_test(self, config_name: str, custom_params: Dict = None, 
                       data_file: str = None, chunk_index: int = None) -> subprocess.Popen:
        """
        Launch a single test in a subprocess
        
        Args:
            config_name: Configuration to use
            custom_params: Custom parameters
            data_file: Data file to use
            chunk_index: Specific chunk to test
            
        Returns:
            Subprocess object
        """
        if config_name not in self.configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        # Generate unique run ID
        run_id = generate_run_id(f"{config_name}_parallel")
        
        # Build command
        cmd = [sys.executable, "run_rolling_window_test.py", 
               "--config", config_name,
               "--run-id", run_id]
        
        if data_file:
            cmd.extend(["--data-file", data_file])
        
        if chunk_index is not None:
            cmd.extend(["--chunk-index", str(chunk_index)])
        
        # Launch subprocess
        print(f"🚀 Launching {config_name} test with run ID: {run_id}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Store process info
        self.processes.append({
            'config_name': config_name,
            'run_id': run_id,
            'process': process,
            'start_time': time.time()
        })
        
        return process
    
    def run_parallel_tests(self, test_configs: List[Dict], max_parallel: int = 4) -> Dict:
        """
        Run multiple tests in parallel
        
        Args:
            test_configs: List of test configurations
            max_parallel: Maximum number of parallel processes
            
        Returns:
            Dictionary with results
        """
        print(f"🔄 PARALLEL TEST RUNNER")
        print(f"📊 Total tests: {len(test_configs)}")
        print(f"⚡ Max parallel: {max_parallel}")
        print("="*60)
        
        active_processes = []
        completed_tests = []
        
        for i, test_config in enumerate(test_configs):
            # Wait if we've reached max parallel processes
            while len(active_processes) >= max_parallel:
                # Check for completed processes
                for proc_info in active_processes[:]:
                    if proc_info['process'].poll() is not None:
                        # Process completed
                        completed_tests.append(proc_info)
                        active_processes.remove(proc_info)
                        print(f"✅ {proc_info['config_name']} completed")
                
                if len(active_processes) >= max_parallel:
                    time.sleep(1)  # Wait before checking again
            
            # Launch new test
            config_name = test_config['config_name']
            custom_params = test_config.get('custom_params')
            data_file = test_config.get('data_file')
            chunk_index = test_config.get('chunk_index')
            
            process = self.run_single_test(
                config_name=config_name,
                custom_params=custom_params,
                data_file=data_file,
                chunk_index=chunk_index
            )
            
            active_processes.append(self.processes[-1])
            print(f"📈 Active processes: {len(active_processes)}")
        
        # Wait for remaining processes
        print(f"\n⏳ Waiting for {len(active_processes)} remaining processes...")
        for proc_info in active_processes:
            proc_info['process'].wait()
            completed_tests.append(proc_info)
            print(f"✅ {proc_info['config_name']} completed")
        
        # Collect results
        return self.collect_results(completed_tests)
    
    def collect_results(self, completed_tests: List[Dict]) -> Dict:
        """
        Collect results from completed tests
        
        Args:
            completed_tests: List of completed test information
            
        Returns:
            Dictionary with collected results
        """
        results = {
            'total_tests': len(completed_tests),
            'completed_tests': [],
            'failed_tests': [],
            'summary': {}
        }
        
        for test_info in completed_tests:
            process = test_info['process']
            config_name = test_info['config_name']
            run_id = test_info['run_id']
            
            # Get process output
            stdout, stderr = process.communicate()
            return_code = process.returncode
            
            test_result = {
                'config_name': config_name,
                'run_id': run_id,
                'return_code': return_code,
                'stdout': stdout,
                'stderr': stderr,
                'duration': time.time() - test_info['start_time']
            }
            
            if return_code == 0:
                results['completed_tests'].append(test_result)
                print(f"✅ {config_name} ({run_id}): SUCCESS")
            else:
                results['failed_tests'].append(test_result)
                print(f"❌ {config_name} ({run_id}): FAILED (code {return_code})")
        
        # Generate summary
        results['summary'] = {
            'success_rate': len(results['completed_tests']) / len(completed_tests) * 100,
            'total_duration': sum(t['duration'] for t in completed_tests),
            'avg_duration': sum(t['duration'] for t in completed_tests) / len(completed_tests)
        }
        
        return results
    
    def save_parallel_results(self, results: Dict, output_file: str = None):
        """
        Save parallel test results
        
        Args:
            results: Results dictionary
            output_file: Output file path
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"parallel_test_results_{timestamp}.json"
        
        # Ensure output directory exists
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_file
        
        # Convert to JSON-serializable format
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            else:
                return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert_types(results), f, indent=2)
        
        print(f"📊 Parallel test results saved to: {output_path}")


def main():
    """Main function for parallel test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multiple optimization tests in parallel')
    parser.add_argument('--configs', nargs='+', default=['default', 'quick_test'],
                       help='Configurations to run in parallel')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum number of parallel processes')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Data file to use for all tests')
    parser.add_argument('--chunk-index', type=int, default=None,
                       help='Specific chunk index to test')
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available configurations')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Custom configuration file path')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ParallelTestRunner(config_file=args.config_file)
    
    # List configurations if requested
    if args.list_configs:
        runner.list_configurations()
        return
    
    # Validate configurations
    invalid_configs = [config for config in args.configs if config not in runner.configs]
    if invalid_configs:
        print(f"❌ Invalid configurations: {invalid_configs}")
        print("Available configurations:")
        runner.list_configurations()
        return
    
    # Create test configurations
    test_configs = []
    for config_name in args.configs:
        test_config = {
            'config_name': config_name,
            'data_file': args.data_file,
            'chunk_index': args.chunk_index
        }
        test_configs.append(test_config)
    
    # Run parallel tests
    results = runner.run_parallel_tests(test_configs, max_parallel=args.max_parallel)
    
    # Save results
    runner.save_parallel_results(results)
    
    # Print summary
    print(f"\n📊 PARALLEL TEST SUMMARY:")
    print(f"Total tests: {results['total_tests']}")
    print(f"Successful: {len(results['completed_tests'])}")
    print(f"Failed: {len(results['failed_tests'])}")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    print(f"Total duration: {results['summary']['total_duration']:.1f}s")
    print(f"Average duration: {results['summary']['avg_duration']:.1f}s")


if __name__ == "__main__":
    main() 