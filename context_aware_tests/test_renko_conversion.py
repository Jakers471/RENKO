#!/usr/bin/env python3
"""
Test script to verify Renko conversion functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import DataLoader
from data.renko_converter import RenkoConverter

def test_renko_conversion():
    """Test Renko conversion with sample data"""
    
    print("🧪 TESTING RENKO CONVERSION")
    print("="*50)
    
    # Initialize components
    data_loader = DataLoader()
    renko_converter = RenkoConverter()
    
    # Get available data files
    available_files = data_loader.get_available_data_files()
    
    if not available_files:
        print("❌ No data files found!")
        return
    
    print(f"Available data files: {available_files}")
    
    # Load first available file
    data_file = available_files[0]
    print(f"\n📊 Loading data: {data_file}")
    
    try:
        data = data_loader.load_csv_data(data_file)
        print(f"✅ Loaded {len(data)} bars")
        
        # Validate data
        if not data_loader.validate_data(data):
            print("❌ Data validation failed")
            return
        
        # Test ATR-based Renko conversion
        print("\n🔄 Testing ATR-based Renko conversion...")
        renko_data = renko_converter.convert_to_renko(
            data=data,
            use_atr=True,
            atr_period=14,
            atr_multiplier=1.0
        )
        
        if not renko_data.empty:
            print("✅ ATR-based conversion successful!")
            
            # Get statistics
            stats = renko_converter.get_renko_statistics(renko_data)
            print(f"\n📊 Renko Statistics:")
            print(f"  Total bricks: {stats['total_bricks']}")
            print(f"  Up bricks: {stats['up_bricks']} ({stats['up_percentage']:.1f}%)")
            print(f"  Down bricks: {stats['down_bricks']} ({stats['down_percentage']:.1f}%)")
            print(f"  Avg brick size: {stats['avg_brick_size']:.4f}")
            print(f"  Brick size range: {stats['min_brick_size']:.4f} - {stats['max_brick_size']:.4f}")
            
            # Show first few bricks
            print(f"\n🔍 First 5 Renko bricks:")
            print(renko_data[['timestamp', 'renko_open', 'renko_close', 'direction', 'brick_count']].head())
            
        else:
            print("❌ ATR-based conversion failed")
            return
        
        # Test fixed-size Renko conversion
        print("\n🔄 Testing fixed-size Renko conversion...")
        fixed_brick_size = data['close'].mean() * 0.01  # 1% of average price
        renko_data_fixed = renko_converter.convert_to_renko(
            data=data,
            use_atr=False,
            brick_size=fixed_brick_size
        )
        
        if not renko_data_fixed.empty:
            print("✅ Fixed-size conversion successful!")
            
            # Get statistics
            stats_fixed = renko_converter.get_renko_statistics(renko_data_fixed)
            print(f"\n📊 Fixed-size Renko Statistics:")
            print(f"  Total bricks: {stats_fixed['total_bricks']}")
            print(f"  Up bricks: {stats_fixed['up_bricks']} ({stats_fixed['up_percentage']:.1f}%)")
            print(f"  Down bricks: {stats_fixed['down_bricks']} ({stats_fixed['down_percentage']:.1f}%)")
            print(f"  Brick size: {fixed_brick_size:.4f}")
            
        else:
            print("❌ Fixed-size conversion failed")
            return
        
        print("\n✅ All Renko conversion tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_renko_conversion() 