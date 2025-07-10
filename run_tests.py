#!/usr/bin/env python3
"""
Test Runner for Trading Backtesting System
"""

import sys
import os
import subprocess

def run_test(test_name):
    """Run a specific test"""
    print(f"\n{'='*50}")
    print(f"Running: {test_name}")
    print(f"{'='*50}")
    
    try:
        if test_name == "rthe":
            subprocess.run([sys.executable, "tests/test_rthe.py"], check=True)
        elif test_name == "debug":
            subprocess.run([sys.executable, "tests/debug_backtest.py"], check=True)
        elif test_name == "csv":
            subprocess.run([sys.executable, "tests/test_csv_merger.py"], check=True)
        elif test_name == "validation":
            subprocess.run([sys.executable, "engine/validation_suite.py"], check=True)
        elif test_name == "interactive":
            subprocess.run([sys.executable, "tests/interactive_rthe_test.py"], check=True)
        else:
            print(f"Unknown test: {test_name}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed with error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [test_name]")
        print("\nAvailable tests:")
        print("  rthe       - R.T.H.E. strategy test")
        print("  debug      - Debug backtest")
        print("  csv        - CSV merger test")
        print("  validation - Full validation suite")
        print("  interactive - Interactive R.T.H.E. tester (NEW!)")
        print("  all        - Run all tests")
        return
    
    test_name = sys.argv[1].lower()
    
    if test_name == "all":
        tests = ["rthe", "debug", "csv", "validation"]
        for test in tests:
            success = run_test(test)
            if not success:
                print(f"\n❌ Test '{test}' failed!")
                break
            print(f"\n✅ Test '{test}' completed successfully!")
    else:
        success = run_test(test_name)
        if success:
            print(f"\n✅ Test '{test_name}' completed successfully!")
        else:
            print(f"\n❌ Test '{test_name}' failed!")

if __name__ == "__main__":
    main() 