# Parallel Testing System

A comprehensive system for running multiple optimization tests simultaneously with isolated outputs and configurable parameters.

## 📁 **Directory Structure**

```
parallel_testing/
├── configs/
│   └── test_configurations.json    # Test configurations (JSON)
├── scripts/
│   ├── run_parallel_tests.py       # Main parallel test runner
│   └── quick_test.py              # Quick validation script
├── examples/
│   └── example_parallel_runs.py   # Example usage patterns
├── results/                       # Parallel test results
└── README.md                      # This file
```

## 🚀 **Quick Start**

### **1. List Available Configurations**
```bash
cd context_aware_tests
python parallel_testing/scripts/run_parallel_tests.py --list-configs
```

### **2. Run Quick Validation**
```bash
python parallel_testing/scripts/quick_test.py
```

### **3. Run Parallel Tests**
```bash
# Run 2 configurations simultaneously
python parallel_testing/scripts/run_parallel_tests.py --configs quick_test renko_coarse --max-parallel 2

# Run comprehensive test
python parallel_testing/scripts/run_parallel_tests.py --configs default renko_fine --chunk-index 0 --max-parallel 2
```

## ⚙️ **Available Configurations**

| Configuration | Combinations | Est. Duration | Description |
|---------------|--------------|---------------|-------------|
| `quick_test` | 1 | 1 min | Fast validation test |
| `renko_coarse` | 4 | 3 min | Quick Renko testing |
| `renko_fine` | 18 | 8 min | Detailed Renko analysis |
| `default` | 144 | 30 min | Standard optimization |
| `renko_ultra_fine` | 600 | 120 min | Comprehensive Renko |
| `ohlc_comprehensive` | 840 | 180 min | Extended OHLC testing |

## 🔧 **Configuration Management**

### **Modify Test Parameters**
Edit `configs/test_configurations.json` to customize:

```json
{
  "configurations": {
    "my_custom_config": {
      "name": "My Custom Test",
      "description": "Custom parameter optimization",
      "atr_lookbacks": [10, 14, 21],
      "adx_lookbacks": [7, 14],
      "ma_lengths": [[20, 50, 200]],
      "range_lookbacks": [20, 30],
      "window_size": 1500,
      "step_size": 500,
      "use_renko": true,
      "renko_atr_period": 14,
      "renko_atr_multiplier": 1.0,
      "estimated_combinations": 12,
      "estimated_duration_minutes": 6
    }
  }
}
```

### **Use Custom Configuration File**
```bash
python parallel_testing/scripts/run_parallel_tests.py --config-file my_configs.json --configs my_custom_config
```

## 📊 **Usage Examples**

### **Example 1: Quick Validation**
```bash
python parallel_testing/scripts/quick_test.py
```

### **Example 2: Renko Comparison**
```bash
python parallel_testing/scripts/run_parallel_tests.py --configs renko_fine renko_coarse --max-parallel 2
```

### **Example 3: OHLC vs Renko**
```bash
python parallel_testing/scripts/run_parallel_tests.py --configs default renko_fine --chunk-index 0 --max-parallel 2
```

### **Example 4: Multiple Chunks**
```bash
python parallel_testing/scripts/run_parallel_tests.py --configs quick_test --chunk-index 0 1 2 --max-parallel 3
```

### **Example 5: Comprehensive Test**
```bash
python parallel_testing/scripts/run_parallel_tests.py --configs default renko_fine renko_coarse quick_test --max-parallel 4
```

## 🛡️ **Safety Features**

### **✅ Unique Output Per Run**
- Each run gets unique ID: `run_20250711_143022_a1b2c3d4`
- Isolated directories: `results/run_20250711_143022_a1b2c3d4/`
- No file conflicts between parallel runs

### **✅ No Shared Global State**
- Each subprocess is fully isolated
- Independent parameter sets and configurations
- No shared caches or variables

### **✅ Configurable Parallelism**
- Control max parallel processes: `--max-parallel 4`
- Automatic process management
- Resource usage optimization

## 📈 **Output Structure**

```
parallel_testing/results/
├── parallel_test_results_20250711_143024.json    # Parallel run summary
└── run_20250711_143022_a1b2c3d4/                 # Individual run results
    ├── grid_search_results_run_20250711_143022_a1b2c3d4_143022.csv
    ├── grid_search_results_run_20250711_143022_a1b2c3d4_143022_top10.json
    └── run_metadata_run_20250711_143022_a1b2c3d4.json
```

## 🔍 **Monitoring and Debugging**

### **Real-time Progress**
- Live process status updates
- Completion notifications
- Duration tracking

### **Error Handling**
- Failed test detection
- Error output capture
- Success rate calculation

### **Result Analysis**
- Individual test results
- Overall summary statistics
- Performance metrics

## 🎯 **Best Practices**

### **1. Start Small**
```bash
# Begin with quick validation
python parallel_testing/scripts/quick_test.py
```

### **2. Scale Gradually**
```bash
# Start with 2 parallel processes
python parallel_testing/scripts/run_parallel_tests.py --configs quick_test renko_coarse --max-parallel 2

# Increase as needed
python parallel_testing/scripts/run_parallel_tests.py --configs default renko_fine --max-parallel 4
```

### **3. Monitor Resources**
- Watch CPU and memory usage
- Adjust `--max-parallel` based on system capacity
- Use `--chunk-index` for targeted testing

### **4. Customize Configurations**
- Modify `test_configurations.json` for your needs
- Add new configurations without code changes
- Estimate duration and combinations

## 🚨 **Troubleshooting**

### **Common Issues**

**1. Configuration Not Found**
```bash
# List available configurations
python parallel_testing/scripts/run_parallel_tests.py --list-configs
```

**2. Process Hangs**
```bash
# Reduce parallel processes
python parallel_testing/scripts/run_parallel_tests.py --max-parallel 1
```

**3. Memory Issues**
```bash
# Use smaller configurations
python parallel_testing/scripts/run_parallel_tests.py --configs quick_test renko_coarse
```

**4. File Path Issues**
```bash
# Ensure you're in the correct directory
cd context_aware_tests
python parallel_testing/scripts/run_parallel_tests.py --configs quick_test
```

## 📝 **Advanced Usage**

### **Custom Configuration Files**
Create your own configuration file:
```json
{
  "configurations": {
    "my_test": {
      "name": "My Test",
      "description": "Custom test configuration",
      "atr_lookbacks": [14],
      "adx_lookbacks": [14],
      "ma_lengths": [[20, 50, 200]],
      "range_lookbacks": [30],
      "window_size": 1000,
      "step_size": 500,
      "use_renko": false,
      "renko_atr_period": 14,
      "renko_atr_multiplier": 1.0,
      "estimated_combinations": 1,
      "estimated_duration_minutes": 1
    }
  }
}
```

### **Batch Processing**
```bash
# Run multiple configurations in sequence
for config in quick_test renko_coarse renko_fine; do
    python parallel_testing/scripts/run_parallel_tests.py --configs $config --max-parallel 1
done
```

## 🎉 **Success Indicators**

- ✅ All processes complete successfully
- ✅ Results saved to isolated directories
- ✅ No file conflicts or overwrites
- ✅ Performance metrics within expected ranges
- ✅ Success rate > 95%

---

**The parallel testing system is now ready for production use!** 🚀 