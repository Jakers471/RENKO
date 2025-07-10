#!/usr/bin/env python3
"""
Quick Command Menu for R.T.H.E. Trading System
Run with: python quick_menu.py
"""

import os
import sys
import subprocess
import time

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(command, description=""):
    """Run a command and show description"""
    print(f"\n🚀 Running: {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        # Run command in the same terminal window
        result = subprocess.run(command, shell=True, check=True, text=True)
        print("✅ Command completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Command interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    input("\nPress Enter to return to menu...")

def show_main_menu():
    """Display the main quick menu"""
    clear_screen()
    print("=" * 60)
    print("🚀 SYSTEM V1 - QUICK COMMAND MENU")
    print("=" * 60)
    print()
    print("📊 SYSTEM V1 TESTING:")
    print("1.  Interactive System V1 Tester")
    print("2.  Parameter Optimization Sweep")
    print("3.  Single Strategy Test")
    print("4.  Trade Visualization")
    print()
    print("🔬 VALIDATION TESTS:")
    print("5.  Monte Carlo Robustness Test")
    print("6.  Walk Forward Analysis")
    print("7.  Comprehensive Validation")
    print("8.  Statistical Significance Test")
    print()
    print("📁 DATA & UTILITIES:")
    print("9.  Refresh Bitcoin Data")
    print("10. Show Available Timeframes")
    print("11. Testing Framework Documentation")
    print()
    print("⚙️  SYSTEM:")
    print("12. Clear Terminal")
    print("13. Show Project Status")
    print("14. Exit")
    print()
    print("=" * 60)

def run_interactive_tester():
    """Run the interactive R.T.H.E. tester"""
    print("\n🚀 Launching Interactive R.T.H.E. Strategy Tester...")
    print("=" * 60)
    run_command("python tests/interactive_rthe_test.py", "Interactive R.T.H.E. Strategy Tester")

def run_parameter_sweep():
    """Run parameter optimization sweep"""
    print("\n🔧 PARAMETER SWEEP MODE")
    print("This will launch the interactive tester")
    print("Select option 3 (Parameter Sweep) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Parameter Optimization Sweep")

def run_single_test():
    """Run single strategy test"""
    print("\n🔧 SINGLE STRATEGY TEST")
    print("This will launch the interactive tester")
    print("Select option 2 (Single Test) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Single Strategy Test")

def run_trade_visualization():
    """Run trade visualization"""
    print("\n🔧 TRADE VISUALIZATION")
    print("This will launch the interactive tester")
    print("Select option 4 (Plot Renko Chart) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Trade Visualization")

def run_monte_carlo():
    """Run Monte Carlo test"""
    print("\n🔧 MONTE CARLO TEST")
    print("This will launch the interactive tester")
    print("Select option 8 (Monte Carlo Test) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Monte Carlo Robustness Test")

def run_walk_forward():
    """Run walk forward analysis"""
    print("\n🔧 WALK FORWARD ANALYSIS")
    print("This will launch the interactive tester")
    print("Select option 9 (Walk Forward Test) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Walk Forward Analysis")

def run_comprehensive_validation():
    """Run comprehensive validation"""
    print("\n🔧 COMPREHENSIVE VALIDATION")
    print("This will launch the interactive tester")
    print("Select option 10 (Walk Forward Monte Carlo) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Comprehensive Validation")

def run_statistical_test():
    """Run statistical significance test"""
    print("\n🔧 STATISTICAL SIGNIFICANCE TEST")
    print("This will launch the interactive tester")
    print("Select option 11 (In-Sample Excellence) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Statistical Significance Test")

def refresh_bitcoin_data():
    """Refresh Bitcoin data"""
    run_command("python utils/merge_bitcoin_data.py", "Refresh Bitcoin Data")

def show_timeframes():
    """Show available timeframes"""
    print("\n📊 AVAILABLE TIMEFRAMES")
    print("This will launch the interactive tester")
    print("Select option 6 (Show Timeframes) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Show Available Timeframes")

def show_testing_framework():
    """Show testing framework documentation"""
    print("\n📚 TESTING FRAMEWORK DOCUMENTATION")
    print("This will launch the interactive tester")
    print("Select option 7 (Testing Framework) in the interactive menu")
    input("Press Enter to continue...")
    run_command("python tests/interactive_rthe_test.py", "Testing Framework Documentation")

def clear_terminal():
    """Clear the terminal"""
    clear_screen()
    print("✅ Terminal cleared!")

def show_project_status():
    """Show project status and file information"""
    clear_screen()
    print("📊 PROJECT STATUS")
    print("=" * 50)
    
    # Check key files
    key_files = [
        "tests/interactive_rthe_test.py",
        "strategies/rthe_strategy.py",
        "engine/backtest_engine.py",
        "utils/data_handler.py",
        "data/bitcoin_merged_data.csv"
    ]
    
    print("\n📁 Key Files Status:")
    for file_path in key_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Check data directory
    print("\n📊 Data Directory:")
    if os.path.exists("data"):
        data_files = [f for f in os.listdir("data") if f.endswith('.csv')]
        print(f"Found {len(data_files)} CSV files:")
        for file in data_files[:5]:  # Show first 5
            print(f"  📄 {file}")
        if len(data_files) > 5:
            print(f"  ... and {len(data_files) - 5} more")
    else:
        print("❌ data/ directory not found")
    
    # Git status
    print("\n🔧 Git Status:")
    try:
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  Uncommitted changes detected")
            print("Modified files:")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"  📝 {line}")
        else:
            print("✅ Working directory clean")
    except:
        print("❌ Git not available or not initialized")
    
    input("\nPress Enter to continue...")

def main():
    """Main menu loop"""
    while True:
        show_main_menu()
        
        try:
            choice = input("Select option (1-14): ").strip()
            
            if choice == '1':
                run_interactive_tester()
            elif choice == '2':
                run_parameter_sweep()
            elif choice == '3':
                run_single_test()
            elif choice == '4':
                run_trade_visualization()
            elif choice == '5':
                run_monte_carlo()
            elif choice == '6':
                run_walk_forward()
            elif choice == '7':
                run_comprehensive_validation()
            elif choice == '8':
                run_statistical_test()
            elif choice == '9':
                refresh_bitcoin_data()
            elif choice == '10':
                show_timeframes()
            elif choice == '11':
                show_testing_framework()
            elif choice == '12':
                clear_terminal()
            elif choice == '13':
                show_project_status()
            elif choice == '14':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-14.")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main() 