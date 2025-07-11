"""
Example Parallel Test Runs
Demonstrates different ways to use the parallel testing system
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from parallel_testing.scripts.run_parallel_tests import ParallelTestRunner

def example_1_quick_validation():
    """Example 1: Quick validation test"""
    print("🔍 EXAMPLE 1: Quick Validation Test")
    print("="*50)
    
    runner = ParallelTestRunner()
    
    # Run quick test on chunk 0
    test_configs = [
        {'config_name': 'quick_test', 'chunk_index': 0}
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=1)
    print(f"✅ Quick validation completed: {len(results['completed_tests'])} successful")
    return results

def example_2_renko_comparison():
    """Example 2: Compare different Renko configurations"""
    print("\n🔍 EXAMPLE 2: Renko Configuration Comparison")
    print("="*50)
    
    runner = ParallelTestRunner()
    
    # Compare fine vs coarse Renko settings
    test_configs = [
        {'config_name': 'renko_fine', 'chunk_index': 0},
        {'config_name': 'renko_coarse', 'chunk_index': 0}
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=2)
    print(f"✅ Renko comparison completed: {len(results['completed_tests'])} successful")
    return results

def example_3_mixed_data_types():
    """Example 3: Compare OHLC vs Renko on same chunk"""
    print("\n🔍 EXAMPLE 3: OHLC vs Renko Comparison")
    print("="*50)
    
    runner = ParallelTestRunner()
    
    # Compare OHLC vs Renko on same data
    test_configs = [
        {'config_name': 'default', 'chunk_index': 0},  # OHLC
        {'config_name': 'renko_fine', 'chunk_index': 0}  # Renko
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=2)
    print(f"✅ Data type comparison completed: {len(results['completed_tests'])} successful")
    return results

def example_4_multiple_chunks():
    """Example 4: Test multiple chunks with same configuration"""
    print("\n🔍 EXAMPLE 4: Multiple Chunks Test")
    print("="*50)
    
    runner = ParallelTestRunner()
    
    # Test same config on different chunks
    test_configs = [
        {'config_name': 'quick_test', 'chunk_index': 0},
        {'config_name': 'quick_test', 'chunk_index': 1},
        {'config_name': 'quick_test', 'chunk_index': 2}
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=3)
    print(f"✅ Multiple chunks test completed: {len(results['completed_tests'])} successful")
    return results

def example_5_comprehensive_test():
    """Example 5: Comprehensive test with multiple configurations"""
    print("\n🔍 EXAMPLE 5: Comprehensive Test")
    print("="*50)
    
    runner = ParallelTestRunner()
    
    # Test multiple configurations on chunk 0
    test_configs = [
        {'config_name': 'default', 'chunk_index': 0},
        {'config_name': 'renko_fine', 'chunk_index': 0},
        {'config_name': 'renko_coarse', 'chunk_index': 0},
        {'config_name': 'quick_test', 'chunk_index': 0}
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=4)
    print(f"✅ Comprehensive test completed: {len(results['completed_tests'])} successful")
    return results

def main():
    """Run all examples"""
    print("🚀 PARALLEL TESTING EXAMPLES")
    print("="*60)
    
    # Run examples
    results = []
    
    try:
        results.append(example_1_quick_validation())
        results.append(example_2_renko_comparison())
        results.append(example_3_mixed_data_types())
        results.append(example_4_multiple_chunks())
        results.append(example_5_comprehensive_test())
        
        # Summary
        total_tests = sum(len(r['completed_tests']) + len(r['failed_tests']) for r in results)
        total_success = sum(len(r['completed_tests']) for r in results)
        total_failed = sum(len(r['failed_tests']) for r in results)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"Total tests run: {total_tests}")
        print(f"Successful: {total_success}")
        print(f"Failed: {total_failed}")
        print(f"Success rate: {total_success/total_tests*100:.1f}%")
        
        print(f"\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 