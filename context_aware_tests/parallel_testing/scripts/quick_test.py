"""
Quick Test Script for Parallel Testing System
Runs a fast validation test to ensure everything is working
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from parallel_testing.scripts.run_parallel_tests import ParallelTestRunner

def run_quick_validation():
    """Run a quick validation test"""
    print("🚀 QUICK VALIDATION TEST")
    print("="*40)
    
    # Initialize runner
    runner = ParallelTestRunner()
    
    # List available configurations
    print("📋 Available configurations:")
    runner.list_configurations()
    
    # Run quick test
    print("\n🔄 Running quick test...")
    test_configs = [
        {'config_name': 'quick_test', 'chunk_index': 0}
    ]
    
    results = runner.run_parallel_tests(test_configs, max_parallel=1)
    
    # Print results
    print(f"\n📊 Quick Test Results:")
    print(f"Success: {len(results['completed_tests'])}")
    print(f"Failed: {len(results['failed_tests'])}")
    
    if results['completed_tests']:
        print("✅ Quick test completed successfully!")
        return True
    else:
        print("❌ Quick test failed!")
        return False

def main():
    """Main function"""
    success = run_quick_validation()
    
    if success:
        print("\n🎉 Parallel testing system is ready!")
        print("You can now run:")
        print("  python parallel_testing/scripts/run_parallel_tests.py --list-configs")
        print("  python parallel_testing/scripts/run_parallel_tests.py --configs quick_test renko_coarse")
    else:
        print("\n⚠️  Please check your setup and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 