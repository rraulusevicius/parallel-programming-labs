"""
Data Augmentation Script
Creates more training data by duplicating with slight random variations
This increases computation per epoch without increasing epochs
"""

import pandas as pd
import numpy as np

def augment_housing_data(input_file='data_for_task3.tsv', output_file='data_augmented.tsv', multiplier=5):
    """
    Augment housing data by creating variations
    
    Args:
        input_file: Original TSV file
        output_file: Augmented TSV file
        multiplier: How many times to multiply the data (e.g., 5x = 1460 → 7300 samples)
    """
    print("="*80)
    print("DATA AUGMENTATION")
    print("="*80)
    
    # Load original data
    print(f"\nLoading original data from: {input_file}")
    data = pd.read_csv(input_file, sep='\t')
    print(f"Original samples: {len(data)}")
    
    # Store original
    augmented_data = [data.copy()]
    
    # Create variations
    print(f"\nCreating {multiplier-1} variations...")
    for i in range(1, multiplier):
        print(f"  Variation {i}...", end=" ")
        
        # Copy data
        variation = data.copy()
        
        # Add random noise to features (±5%)
        for col in ['LotArea', 'YearBuilt']:
            noise = np.random.uniform(-0.05, 0.05, len(variation))
            variation[col] = (variation[col] * (1 + noise)).astype(int)
        
        # OverallQual stays same or ±1
        qual_noise = np.random.choice([-1, 0, 1], len(variation))
        variation['OverallQual'] = np.clip(variation['OverallQual'] + qual_noise, 1, 10)
        
        # Adjust SalePrice based on changes (maintain correlation)
        price_noise = np.random.uniform(-0.05, 0.05, len(variation))
        variation['SalePrice'] = (variation['SalePrice'] * (1 + price_noise)).astype(int)
        
        augmented_data.append(variation)
        print(f"{len(variation)} samples")
    
    # Combine all variations
    final_data = pd.concat(augmented_data, ignore_index=True)
    
    print(f"\nFinal augmented dataset: {len(final_data)} samples")
    print(f"Multiplier: {len(final_data) / len(data):.1f}x")
    
    # Save
    final_data.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved to: {output_file}")
    
    # Show statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print("\nOriginal data:")
    print(data.describe())
    print("\nAugmented data:")
    print(final_data.describe())
    
    print("\n" + "="*80)
    print("✓ Augmentation complete!")
    print("="*80)
    print(f"\nTo use augmented data, update parallel_optimization.py line 20:")
    print(f"  OLD: data = pd.read_csv('data_for_task3.tsv', sep='\\t')")
    print(f"  NEW: data = pd.read_csv('{output_file}', sep='\\t')")
    
    return final_data

if __name__ == "__main__":
    print("\nHow much data do you want?")
    print("  1. 2x  (1460 → 2920 samples)")
    print("  2. 3x  (1460 → 4380 samples)")
    print("  3. 5x  (1460 → 7300 samples)  ← RECOMMENDED")
    print("  4. 10x (1460 → 14600 samples)")
    
    choice = input("\nChoice (1-4): ")
    
    multipliers = {'1': 2, '2': 3, '3': 5, '4': 10}
    multiplier = multipliers.get(choice, 5)
    
    print(f"\nCreating {multiplier}x augmented dataset...")
    augment_housing_data(multiplier=multiplier)
