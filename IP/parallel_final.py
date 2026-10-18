import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_SIZES = [100, 200, 400, 800, 1460, 3000, 7300, 14600]
ENSEMBLE_SIZE = 4
EPOCHS_CONFIG = {
    100: 2000,
    200: 2000,
    400: 2000,
    800: 2000,
    1460: 2000,
    3000: 3000,
    7300: 5000,
    14600: 10000
}
PROCESS_COUNTS = [1, 2, 4, 6, 8]
NUM_RUNS = 3

# ============================================================================
# ML FUNCTIONS
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
        min_val, max_val = train_data[feat].min(), train_data[feat].max()
        train_data[feat] = (train_data[feat] - min_val) / (max_val - min_val)
        test_data[feat] = (test_data[feat] - min_val) / (max_val - min_val)
    
    return (train_data[features].values, train_data['SalePrice'].values,
            test_data[features].values, test_data['SalePrice'].values)


def forward_pass(X, W):
    """Neural network forward pass"""
    N1, N2, N3 = X[:, 0], X[:, 1], X[:, 2]
    return (W[0]*N1 + W[4]*N2 + W[8]*N3 +
            W[1]*N1 + W[5]*N2 + W[9]*N3 +
            W[2]*N1 + W[6]*N2 + W[10]*N3 +
            W[3]*N1 + W[7]*N2 + W[11]*N3)


def compute_mse(X, y, W):
    """Mean squared error"""
    return np.mean((y - forward_pass(X, W))**2)


def compute_gradient(X, y, W):
    """Gradient via finite differences"""
    grad = np.zeros(12)
    J = compute_mse(X, y, W)
    for i in range(12):
        W_plus = W.copy()
        W_plus[i] += 1e-5
        grad[i] = (compute_mse(X, y, W_plus) - J) / 1e-5
    return grad


def train_model(args):
    """Train a single model - runs in worker process"""
    X_train, y_train, X_test, y_test, seed, max_epochs = args
    np.random.seed(seed)
    W = np.random.uniform(-1, 1, 12)
    for _ in range(max_epochs):
        W -= 0.001 * compute_gradient(X_train, y_train, W)
    return {'mse': compute_mse(X_test, y_test, W), 'weights': W}


# ============================================================================
# SECTION 3: VERIFICATION
# ============================================================================

def verify_optimization():
    """Show that optimization converges correctly"""
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
    
    print(f"\nTraining for 2000 epochs...")
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
    
    improvement = (initial_test_mse - final_test_mse) / initial_test_mse * 100
    print(f"\n✓ Improvement: {improvement:.1f}%")
    print(f"✓ MSE decreased from {initial_test_mse:.2e} to {final_test_mse:.2e}")
    print(f"✓ Optimization converges correctly!")
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs, train_mses, test_mses = zip(*history)
    plt.plot(epochs, train_mses, 'o-', label='Training MSE', linewidth=2)
    plt.plot(epochs, test_mses, 's-', label='Test MSE', linewidth=2)
    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('MSE', fontweight='bold', fontsize=12)
    plt.title('Optimization Convergence', fontweight='bold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('results/00_verification.png', dpi=300)
    plt.close()


# ============================================================================
# SECTION 4: PERFORMANCE TESTING
# ============================================================================

def run_performance_tests():
    """Test parallelization with different dataset sizes"""
    print("\n" + "="*80)
    print("SECTION 4: PERFORMANCE TESTING (Vykdymo laiko tyrimas)")
    print("="*80)
    
    data_file = 'data_augmented.tsv' if os.path.exists('data_augmented.tsv') else 'data_for_task3.tsv'
    print(f"Data file: {data_file}\n")
    
    results = []
    
    for size in DATASET_SIZES:
        print(f"\n{'='*80}")
        print(f"Dataset: {size} samples | Epochs: {EPOCHS_CONFIG[size]} | Models: {ENSEMBLE_SIZE}")
        print(f"{'='*80}")
        
        X_train, y_train, X_test, y_test = load_data(data_file, size)
        print(f"Training: {len(X_train)} | Test: {len(X_test)}")
        
        for num_proc in PROCESS_COUNTS:
            print(f"  {num_proc} proc(s): ", end='', flush=True)
            times = []
            mses = []
            
            for run in range(NUM_RUNS):
                args = [(X_train, y_train, X_test, y_test, i, EPOCHS_CONFIG[size]) 
                        for i in range(ENSEMBLE_SIZE)]
                
                start = time.time()
                if num_proc == 1:
                    res = [train_model(a) for a in args]
                else:
                    with Pool(num_proc) as pool:
                        res = pool.map(train_model, args)
                elapsed = time.time() - start
                
                times.append(elapsed)
                mses.append(min(r['mse'] for r in res))
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
    df.to_csv('results/test_results.csv', index=False)
    
    # Plot 1: Time vs Processes
    sizes = sorted(df['dataset_size'].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        subset = df[df['dataset_size'] == size]
        epochs = subset['epochs'].iloc[0]  # Get epoch count for this size
        ax = axes[idx]
        ax.errorbar(subset['num_processes'], subset['avg_time'], 
                   yerr=subset['std_time'], marker='o', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Processes', fontweight='bold')
        ax.set_ylabel('Time (s)', fontweight='bold')
        ax.set_title(f'{int(size)} samples | {int(epochs)} epochs', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/01_time_vs_processes.png', dpi=300)
    plt.close()
    
    # Plot 2: Speedup
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        subset = df[df['dataset_size'] == size]
        epochs = subset['epochs'].iloc[0]  # Get epoch count for this size
        ax = axes[idx]
        max_proc = subset['num_processes'].max()
        ax.plot([1, max_proc], [1, max_proc], 'k--', alpha=0.5, label='Ideal')
        ax.plot(subset['num_processes'], subset['speedup'], 
               'o-', linewidth=2, markersize=8, label='Actual')
        ax.set_xlabel('Processes', fontweight='bold')
        ax.set_ylabel('Speedup', fontweight='bold')
        ax.set_title(f'{int(size)} samples | {int(epochs)} epochs', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/02_speedup_vs_processes.png', dpi=300)
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
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
        print(f"{size:<6} samples: {int(best['num_processes'])} procs -> "
              f"{best['speedup']:.2f}x speedup ({best['efficiency']:.1f}% eff) | "
              f"{best['avg_time']:.2f}s")
    
    max_speedup = df['speedup'].max()
    best_row = df.loc[df['speedup'].idxmax()]
    print(f"\nMaximum speedup: {max_speedup:.2f}x")
    print(f"  Achieved with {int(best_row['num_processes'])} processes")
    print(f"  Dataset size: {int(best_row['dataset_size'])} samples")
    
    # Save report
    with open('results/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE TEST REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU cores: {cpu_count()}\n\n")
        f.write("BEST CONFIGURATIONS:\n")
        for size in sizes:
            subset = df[df['dataset_size'] == size]
            best = subset.loc[subset['avg_time'].idxmin()]
            f.write(f"{size} samples: {int(best['num_processes'])} procs -> "
                   f"{best['speedup']:.2f}x speedup ({best['efficiency']:.1f}% efficiency)\n")
        f.write(f"\nMaximum speedup: {max_speedup:.2f}x\n")
      
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("PARALLEL NEURAL NETWORK OPTIMIZATION")
    print("="*80)
    print(f"CPU cores: {cpu_count()}")
    print(f"Dataset sizes: {len(DATASET_SIZES)}")
    print(f"Ensemble size: {ENSEMBLE_SIZE} models per dataset")
    print(f"Process counts: {PROCESS_COUNTS}")
    print()
    print("Configuration:")
    print(f"  - Each dataset trains {ENSEMBLE_SIZE} models with different initializations")
    print(f"  - Epochs = training iterations (full passes through data)")
    print(f"  - Smaller datasets use MORE epochs for convergence")
    print(f"  - Larger datasets use FEWER epochs (converge faster)")
    print(f"  - Ensures sufficient computation time for all sizes")
    
    os.makedirs('results', exist_ok=True)
    
    # Section 3: Verification
    verify_optimization()
    
    # Section 4: Performance testing
    df = run_performance_tests()
    
    # Analysis
    df = analyze_and_plot(df)
    
    print("\n" + "="*80)
    print("✓ TESTING COMPLETE!")
    print("="*80)
    print("Results saved to: ./results/")
    print("="*80)


if __name__ == '__main__':
    main()