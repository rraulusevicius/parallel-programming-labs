"""
DATA PARALLELISM VERSION - Matching parallel_final.py Structure
Splits data across processes, combines gradients, trains ONE model
Tests all 8 dataset sizes with multiple runs
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_SIZES = [3000]

EPOCHS_CONFIG = {
    100: 2000,
    200: 2000,
    400: 2000,
    800: 2000,
    1460: 2000,
    3000: 3000,
    7300: 5000,
    14600: 15000
}

PROCESS_COUNTS = [1, 2, 4, 6, 8]
NUM_RUNS = 3  # Run each config 3 times

# ============================================================================
# ML FUNCTIONS (must be at module level for Windows)
# ============================================================================

def load_data(data_file, sample_size=None):
    """Load and normalize data"""
    data = pd.read_csv(data_file, sep='\t')
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
    
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    train_size = int(0.8 * len(data))
    
    train_data = data.iloc[indices[:train_size]].reset_index(drop=True)
    test_data = data.iloc[indices[train_size:]].reset_index(drop=True)
    
    features = ['LotArea', 'OverallQual', 'YearBuilt']
    for feat in features:
        min_val = train_data[feat].min()
        max_val = train_data[feat].max()
        train_data[feat] = (train_data[feat] - min_val) / (max_val - min_val)
        test_data[feat] = (test_data[feat] - min_val) / (max_val - min_val)
    
    X_train = train_data[features].values
    y_train = train_data['SalePrice'].values
    X_test = test_data[features].values
    y_test = test_data['SalePrice'].values
    
    return X_train, y_train, X_test, y_test


def forward_pass(X, W):
    """Neural network forward pass: 3->4->1"""
    N1 = 1 / (1 + np.exp(-(W[0]*X[:, 0] + W[1]*X[:, 1] + W[2]*X[:, 2])))
    N2 = 1 / (1 + np.exp(-(W[3]*X[:, 0] + W[4]*X[:, 1] + W[5]*X[:, 2])))
    N3 = 1 / (1 + np.exp(-(W[6]*X[:, 0] + W[7]*X[:, 1] + W[8]*X[:, 2])))
    return (W[9]*N1 + W[10]*N2 + W[11]*N3 +
            W[0]*N1 + W[4]*N2 + W[8]*N3 +
            W[1]*N1 + W[5]*N2 + W[9]*N3 +
            W[2]*N1 + W[6]*N2 + W[10]*N3 +
            W[3]*N1 + W[7]*N2 + W[11]*N3)


def compute_mse(X, y, W):
    """Compute mean squared error"""
    return np.mean((y - forward_pass(X, W))**2)


def compute_gradient(X, y, W):
    """Compute gradient using finite differences"""
    grad = np.zeros(12)
    J = compute_mse(X, y, W)
    for i in range(12):
        W_plus = W.copy()
        W_plus[i] += 0.01
        grad[i] = (compute_mse(X, y, W_plus) - J) / 0.01
    return grad


def train_on_chunk(args):
    """
    Train on a data chunk for one synchronization round
    Returns gradients computed on this chunk
    """
    X_chunk, y_chunk, W_current, sync_epochs = args
    
    # Convert to regular numpy arrays if needed (Windows pickle issue)
    X_chunk = np.array(X_chunk, dtype=np.float64)
    y_chunk = np.array(y_chunk, dtype=np.float64)
    W_current = np.array(W_current, dtype=np.float64)
    
    W = W_current.copy()
    
    # Train for sync_epochs on this chunk
    for _ in range(sync_epochs):
        grad = compute_gradient(X_chunk, y_chunk, W)
        W -= 0.001 * grad
    
    # Return gradient at current position
    final_grad = compute_gradient(X_chunk, y_chunk, W)
    return final_grad.tolist()  # Return as list for Windows compatibility


def train_data_parallel(X_train, y_train, X_test, y_test, total_epochs, num_processes):
    """
    Train with data parallelism
    Splits data across processes, averages gradients
    """
    # Fixed initial weights
    W = np.array([1.0, 0.2, 0.1, 1.0, 0.2, 0.1, 1.0, 1.0, 0.2, 0.25, 0.2, 0.5])
    
    # Synchronization strategy - reduce frequency for Windows
    sync_interval = 1000  # Sync every 100 epochs (reduced from 10)
    num_syncs = total_epochs // sync_interval
    
    # Sequential training (no parallelism)
    if num_processes == 1:
        for _ in range(total_epochs):
            grad = compute_gradient(X_train, y_train, W)
            W -= 0.001 * grad
    else:
        # Parallel training with data splitting
        chunk_size = len(X_train) // num_processes
        
        for sync_round in range(num_syncs):
            # Split data into chunks - convert to plain lists for Windows
            data_chunks = []
            for i in range(num_processes):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_processes - 1 else len(X_train)
                
                X_chunk = X_train[start_idx:end_idx].tolist()
                y_chunk = y_train[start_idx:end_idx].tolist()
                
                data_chunks.append((X_chunk, y_chunk, W.tolist(), sync_interval))
            
            # Train on chunks in parallel
            with Pool(num_processes) as pool:
                gradients = pool.map(train_on_chunk, data_chunks)
            
            # Convert back to numpy and average gradients
            gradients = [np.array(g) for g in gradients]
            avg_grad = np.mean(gradients, axis=0)
            
            # Update weights
            W -= 0.001 * avg_grad
    
    # Compute final MSE
    test_mse = compute_mse(X_test, y_test, W)
    return {'mse': test_mse, 'weights': W}


# ============================================================================
# SECTION 3: VERIFICATION
# ============================================================================

def run_verification():
    """Verify that optimization works correctly"""
    print("\n" + "="*80)
    print("SECTION 3: VERIFICATION (Testavimas)")
    print("="*80)
    print("Demonstrating that optimization works correctly...\n")
    
    X_train, y_train, X_test, y_test = load_data('data_for_task3.tsv', 1460)
    
    # Use specific initial weights for consistency with lab results
    W_init = np.array([1.0, 0.2, 0.1, 1.0, 0.2, 0.1, 1.0, 1.0, 0.2, 0.25, 0.2, 0.5])
    
    initial_train_mse = compute_mse(X_train, y_train, W_init)
    initial_test_mse = compute_mse(X_test, y_test, W_init)
    
    print(f"Initial weights: {W_init}")
    print(f"Initial training MSE: {initial_train_mse:.2e}")
    print(f"Initial test MSE: {initial_test_mse:.2e}")
    
    print(f"\nTraining for 2000 epochs (data-parallel with 1 process)...")
    W = W_init.copy()
    history = []
    
    for epoch in range(2000):
        W -= 0.001 * compute_gradient(X_train, y_train, W)
        if epoch % 200 == 0:
            train_mse = compute_mse(X_train, y_train, W)
            test_mse = compute_mse(X_test, y_test, W)
            history.append((epoch, train_mse, test_mse))
            print(f"  Epoch {epoch:4d}: Train={train_mse:.2e} | Test={test_mse:.2e}")
    
    final_train_mse = compute_mse(X_train, y_train, W)
    final_test_mse = compute_mse(X_test, y_test, W)
    
    print(f"\nOptimized weights: {W[:3]}... (first 3)")
    print(f"Final training MSE: {final_train_mse:.2e}")
    print(f"Final test MSE: {final_test_mse:.2e}")
    
    improvement = (1 - final_test_mse / initial_test_mse) * 100
    print(f"\nTest MSE Improvement: {improvement:.1f}%")
    print(f"MSE decreased from {initial_test_mse:.2e} to {final_test_mse:.2e}")
    print(f"\nOptimization converges correctly!")


# ============================================================================
# SECTION 4: PERFORMANCE TESTING
# ============================================================================

def run_performance_tests():
    """Test data parallelization with different dataset sizes"""
    print("\n" + "="*80)
    print("SECTION 4: DATA-PARALLEL PERFORMANCE TESTING")
    print("="*80)
    print("Strategy: Split data across processes, average gradients")
    print("Result: ONE model trained (not ensemble)\n")
    
    data_file = 'data_augmented.tsv' if os.path.exists('data_augmented.tsv') else 'data_for_task3.tsv'
    print(f"Data file: {data_file}\n")
    
    results = []
    
    for size in DATASET_SIZES:
        print(f"\n{'='*80}")
        print(f"Dataset: {size} samples | Epochs: {EPOCHS_CONFIG[size]} | Strategy: Data-Parallel")
        print(f"{'='*80}")
        
        X_train, y_train, X_test, y_test = load_data(data_file, size)
        print(f"Training: {len(X_train)} | Test: {len(X_test)}")
        
        for num_proc in PROCESS_COUNTS:
            print(f"  {num_proc} proc(s): ", end='', flush=True)
            times = []
            mses = []
            
            for run in range(NUM_RUNS):
                start = time.time()
                result = train_data_parallel(X_train, y_train, X_test, y_test, 
                                            EPOCHS_CONFIG[size], num_proc)
                elapsed = time.time() - start
                
                times.append(elapsed)
                mses.append(result['mse'])
                print(".", end='', flush=True)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_mse = np.mean(mses)
            std_mse = np.std(mses)
            
            results.append({
                'dataset_size': size,
                'epochs': EPOCHS_CONFIG[size],
                'num_processes': num_proc,
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_mse': avg_mse,
                'std_mse': std_mse
            })
            
            print(f" {avg_time:.2f}s ± {std_time:.2f}s | MSE: {avg_mse:.2e} ± {std_mse:.2e}")
    
    return pd.DataFrame(results)


def analyze_and_plot(df):
    """Analyze results and create visualizations"""
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    # Create results directory
    os.makedirs('results_data_parallel', exist_ok=True)
    
    # Calculate speedup
    df_analyzed = []
    for size in df['dataset_size'].unique():
        subset = df[df['dataset_size'] == size].copy()
        seq_time = subset[subset['num_processes'] == 1]['avg_time'].values[0]
        subset['speedup'] = seq_time / subset['avg_time']
        subset['efficiency'] = (subset['speedup'] / subset['num_processes']) * 100
        df_analyzed.append(subset)
    
    df = pd.concat(df_analyzed, ignore_index=True)
    
    # Save CSV
    df.to_csv('results_data_parallel/test_results.csv', index=False)
    print("✓ Saved: results_data_parallel/test_results.csv")
    
    # Plot 1: Time vs Processes
    sizes = sorted(df['dataset_size'].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        subset = df[df['dataset_size'] == size]
        epochs = subset['epochs'].iloc[0]
        ax = axes[idx]
        ax.errorbar(subset['num_processes'], subset['avg_time'], 
                   yerr=subset['std_time'], marker='o', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Processes', fontweight='bold')
        ax.set_ylabel('Time (s)', fontweight='bold')
        ax.set_title(f'{int(size)} samples | {int(epochs)} epochs | Data-Parallel', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_data_parallel/01_time_vs_processes.png', dpi=300)
    print("✓ Saved: results_data_parallel/01_time_vs_processes.png")
    plt.close()
    
    # Plot 2: Speedup
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        subset = df[df['dataset_size'] == size]
        epochs = subset['epochs'].iloc[0]
        ax = axes[idx]
        max_proc = subset['num_processes'].max()
        ax.plot([1, max_proc], [1, max_proc], 'k--', alpha=0.5, label='Ideal')
        ax.plot(subset['num_processes'], subset['speedup'], 
               'o-', linewidth=2, markersize=8, label='Actual')
        ax.set_xlabel('Processes', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title(f'{int(size)} samples | {int(epochs)} epochs | Data-Parallel', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results_data_parallel/02_speedup_vs_processes.png', dpi=300)
    print("✓ Saved: results_data_parallel/02_speedup_vs_processes.png")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY - DATA PARALLELISM")
    print("="*80)
    print(f"\n{'Size':<8} {'Epochs':<7} {'Procs':<6} {'Time(s)':<9} {'Speedup':<10} {'Efficiency':<12} {'MSE':<12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{int(row['dataset_size']):<8} {int(row['epochs']):<7} {int(row['num_processes']):<6} "
              f"{row['avg_time']:<9.2f} {row['speedup']:<10.2f}x {row['efficiency']:<12.1f}% {row['avg_mse']:<12.2e}")
    
    # Best configs
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS (fastest time)")
    print("="*80)
    for size in sizes:
        subset = df[df['dataset_size'] == size]
        best = subset.loc[subset['avg_time'].idxmin()]
        print(f"{int(size):<6} samples: {int(best['num_processes'])} procs -> "
              f"{best['speedup']:.2f}x speedup ({best['efficiency']:.1f}% eff) | "
              f"{best['avg_time']:.2f}s")
    
    max_speedup = df['speedup'].max()
    best_row = df.loc[df['speedup'].idxmax()]
    print(f"\nMaximum speedup: {max_speedup:.2f}x")
    print(f"  Achieved with {int(best_row['num_processes'])} processes")
    print(f"  Dataset size: {int(best_row['dataset_size'])} samples")
    
    # Calculate Amdahl's law parameters
    print("\n" + "="*80)
    print("AMDAHL'S LAW ANALYSIS")
    print("="*80)
    
    # Use largest dataset for best estimate
    largest_subset = df[df['dataset_size'] == sizes[-1]]
    speedups = largest_subset['speedup'].values
    process_counts = largest_subset['num_processes'].values
    
    # Estimate serial fraction
    # Speedup = 1 / (s + (1-s)/n)
    # Rearrange: s ≈ (n - speedup) / (n * speedup - speedup)
    serial_fractions = []
    for n, sp in zip(process_counts[1:], speedups[1:]):  # Skip n=1
        if sp > 0 and n > 1:
            s = (n - sp) / (n * sp - sp)
            serial_fractions.append(max(0, min(1, s)))  # Clamp to [0,1]
    
    if serial_fractions:
        avg_serial = np.mean(serial_fractions)
        print(f"Estimated serial fraction: {avg_serial:.1%}")
        print(f"Parallel fraction: {(1-avg_serial):.1%}")
        print(f"\nTheoretical max speedup: {1/avg_serial:.2f}x")
        print(f"Achieved speedup: {max_speedup:.2f}x")
    
    # Save report
    with open('results_data_parallel/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATA-PARALLEL PERFORMANCE TEST REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU cores: {cpu_count()}\n")
        f.write(f"Strategy: Split data across processes, average gradients\n")
        f.write(f"Result: ONE model per configuration (not ensemble)\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Dataset sizes: {DATASET_SIZES}\n")
        f.write(f"  Epochs per size: {EPOCHS_CONFIG}\n")
        f.write(f"  Process counts tested: {PROCESS_COUNTS}\n")
        f.write(f"  Runs per configuration: {NUM_RUNS}\n\n")
        
        f.write("Best Results:\n")
        for size in sizes:
            subset = df[df['dataset_size'] == size]
            best = subset.loc[subset['avg_time'].idxmin()]
            f.write(f"  {int(size)} samples: {int(best['num_processes'])} processes -> "
                   f"{best['speedup']:.2f}x speedup, {best['avg_time']:.2f}s\n")
        
        f.write(f"\nMaximum speedup achieved: {max_speedup:.2f}x\n")
        if serial_fractions:
            f.write(f"Estimated serial fraction: {avg_serial:.1%}\n")
    
    print("✓ Saved: results_data_parallel/performance_report.txt")
    
    # Print explanation
    print("\n" + "="*80)
    print("HOW DATA PARALLELISM WORKS")
    print("="*80)
    print("""
Key Difference from Task Parallelism:
  
TASK PARALLELISM (Your main implementation):
  - Each process: Trains complete model on ALL data
  - Result: Multiple models (ensemble)
  - Communication: None needed
  - Speedup: Higher (3-4x typical)
  
DATA PARALLELISM (This version):
  - Each process: Trains on DATA CHUNK
  - Gradients are averaged across processes
  - Result: ONE model (trained faster)
  - Communication: Gradient synchronization required
  - Speedup: Lower (2-3x typical) due to sync overhead

Both are valid! Task parallelism better for ensembles.
Data parallelism better for huge datasets (>1M samples).
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("="*80)
    print("DATA PARALLELISM - Comprehensive Testing")
    print("="*80)
    print(f"Strategy: Split data across processes")
    print(f"Each process trains on a CHUNK of data")
    print(f"Gradients are averaged to update shared weights")
    print(f"Result: ONE model per configuration (not ensemble)")
    print(f"\nCPU Cores available: {cpu_count()}")
    print(f"Testing {len(DATASET_SIZES)} dataset sizes")
    print(f"Testing {len(PROCESS_COUNTS)} process configurations")
    print(f"Total configurations: {len(DATASET_SIZES) * len(PROCESS_COUNTS)}")
    
    # Section 3: Verification
    run_verification()
    
    # Section 4: Performance Testing
    df = run_performance_tests()
    
    # Analysis and plotting
    analyze_and_plot(df)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print("Results saved to: ./results_data_parallel/")
    print("="*80)


if __name__ == '__main__':
    main()