"""
ENSEMBLE AVERAGING VERSION - Compare Single Best vs. Ensemble Average
Shows the advantage of combining multiple trained models
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import time
from multiprocessing import Pool

# ============================================================================
# CONFIGURATION - All 8 dataset sizes with ensemble averaging
# ============================================================================
DATASET_SIZES = [100, 200, 400, 800, 1460, 3000, 7300, 14600]
ENSEMBLE_SIZE = 8  # Train 8 models per size
EPOCHS = {
    100: 2000,
    200: 2000,
    400: 2000,
    800: 2000,
    1460: 2000,
    3000: 2000,
    7300: 2000,
    14600: 2000
}
PROCESS_COUNTS = [1, 2, 4, 8]  # Test fewer process counts for speed

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(data_file, sample_size):
    """Load and normalize data"""
    data = pd.read_csv(data_file, sep='\t')
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
    """Compute mean squared error"""
    return np.mean((y - forward_pass(X, W))**2)


def compute_gradient(X, y, W):
    """Compute gradient using finite differences"""
    grad = np.zeros(12)
    J = compute_mse(X, y, W)
    for i in range(12):
        W_plus = W.copy()
        W_plus[i] += 1e-5
        grad[i] = (compute_mse(X, y, W_plus) - J) / 1e-5
    return grad


def train_model(args):
    """Train a single model - returns weights AND test MSE"""
    X_train, y_train, X_test, y_test, seed, max_epochs = args
    np.random.seed(seed)
    W = np.random.uniform(-1, 1, 12)
    for _ in range(max_epochs):
        W -= 0.001 * compute_gradient(X_train, y_train, W)
    test_mse = compute_mse(X_test, y_test, W)
    return {'weights': W, 'mse': test_mse}


def ensemble_predict(X, models):
    """Average predictions from all models"""
    predictions = [forward_pass(X, model['weights']) for model in models]
    return np.mean(predictions, axis=0)


def run_test():
    """Main test function"""
    print("="*70)
    print("ENSEMBLE AVERAGING TEST - Single Best vs. Ensemble Average")
    print("="*70)
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Ensemble size: {ENSEMBLE_SIZE} models")
    print(f"Process counts: {PROCESS_COUNTS}")
    print()
    print("Comparison:")
    print("  - Single Best: Pick the best model (current approach)")
    print("  - Ensemble Avg: Average predictions from ALL models")
    print("  - Ensemble avg typically 5-10% better accuracy!")
    print()
    print("Note: Improvement varies based on model diversity.")
    print("      Simple problems may show small improvement.")
    print("      Complex problems with diverse models show larger gains.")
    
    data_file = 'data_augmented.tsv' if os.path.exists('data_augmented.tsv') else 'data_for_task3.tsv'
    print(f"\nData file: {data_file}")
    
    # Store all results
    all_results = []
    
    # For demonstration: show sample predictions
    show_sample_predictions = True
    
    for size in DATASET_SIZES:
        print(f"\n{'='*70}")
        print(f"Dataset: {size} samples | Epochs: {EPOCHS[size]} | Models: {ENSEMBLE_SIZE}")
        print(f"{'='*70}")
        
        X_train, y_train, X_test, y_test = load_data(data_file, size)
        print(f"Training: {len(X_train)} | Test: {len(X_test)}")
        
        seq_time = None
        
        for num_proc in PROCESS_COUNTS:
            print(f"  {num_proc} process(es)... ", end='', flush=True)
            
            args = [(X_train, y_train, X_test, y_test, i, EPOCHS[size]) 
                    for i in range(ENSEMBLE_SIZE)]
            
            start = time.time()
            if num_proc == 1:
                models = [train_model(a) for a in args]
            else:
                with Pool(num_proc) as pool:
                    models = pool.map(train_model, args)
            elapsed = time.time() - start
            
            # Evaluate both approaches
            best_single_mse = min(model['mse'] for model in models)
            
            # Ensemble: average predictions
            ensemble_pred = ensemble_predict(X_test, models)
            ensemble_mse = np.mean((y_test - ensemble_pred)**2)
            
            improvement = (best_single_mse - ensemble_mse) / best_single_mse * 100
            
            if num_proc == 1:
                seq_time = elapsed
                print(f"{elapsed:.2f}s")
                print(f"    Single Best MSE: {best_single_mse:.2e}")
                print(f"    Ensemble Avg MSE: {ensemble_mse:.2e} ({improvement:+.1f}% better)")
                
                # Show sample predictions for first size only
                if show_sample_predictions and size == DATASET_SIZES[0]:
                    print(f"\n    Sample predictions (first 3 test samples):")
                    for i in range(min(3, len(X_test))):
                        individual_preds = [forward_pass(X_test[i:i+1], m['weights'])[0] 
                                          for m in models]
                        ensemble_pred_sample = np.mean(individual_preds)
                        print(f"      Sample {i+1}:")
                        print(f"        Actual:    ${y_test[i]:>12,.0f}")
                        print(f"        Model 1:   ${individual_preds[0]:>12,.0f}")
                        print(f"        Model 2:   ${individual_preds[1]:>12,.0f}")
                        print(f"        Model 8:   ${individual_preds[7]:>12,.0f}")
                        print(f"        Ensemble:  ${ensemble_pred_sample:>12,.0f} (avg of 8)")
                    show_sample_predictions = False  # Only show once
                
                all_results.append({
                    'size': size,
                    'procs': num_proc,
                    'time': elapsed,
                    'best_mse': best_single_mse,
                    'ensemble_mse': ensemble_mse,
                    'improvement': improvement,
                    'speedup': 1.0,
                    'efficiency': 100.0
                })
            else:
                speedup = seq_time / elapsed
                efficiency = (speedup / num_proc) * 100
                print(f"{elapsed:.2f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.1f}%")
                print(f"    Single Best MSE: {best_single_mse:.2e}")
                print(f"    Ensemble Avg MSE: {ensemble_mse:.2e} ({improvement:+.1f}% better)")
                all_results.append({
                    'size': size,
                    'procs': num_proc,
                    'time': elapsed,
                    'best_mse': best_single_mse,
                    'ensemble_mse': ensemble_mse,
                    'improvement': improvement,
                    'speedup': speedup,
                    'efficiency': efficiency
                })
    
    # Summary tables
    print("\n" + "="*70)
    print("PARALLELIZATION RESULTS")
    print("="*70)
    print(f"\n{'Size':<8} {'Procs':<6} {'Time(s)':<9} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['size']:<8} {r['procs']:<6} {r['time']:<9.2f} "
              f"{r['speedup']:<10.2f}x {r['efficiency']:<12.1f}%")
    
    print("\n" + "="*70)
    print("ENSEMBLE AVERAGING BENEFIT")
    print("="*70)
    print(f"\n{'Size':<8} {'Procs':<6} {'Best Single':<14} {'Ensemble Avg':<14} {'Improvement':<12}")
    print("-"*70)
    
    for r in all_results:
        print(f"{r['size']:<8} {r['procs']:<6} {r['best_mse']:<14.2e} "
              f"{r['ensemble_mse']:<14.2e} {r['improvement']:<12.1f}%")
    
    # Average improvement
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"\nParallelization (with {ENSEMBLE_SIZE} models):")
    best_speedup = max(r['speedup'] for r in all_results)
    best_config = [r for r in all_results if r['speedup'] == best_speedup][0]
    print(f"  Best speedup: {best_speedup:.2f}x with {best_config['procs']} processes")
    print(f"  Efficiency: {best_config['efficiency']:.1f}%")
    
    print(f"\nEnsemble Averaging Benefit:")
    if avg_improvement > 1.0:
        print(f"  Average improvement: {avg_improvement:.1f}%")
        print(f"  Ensemble avg consistently better than best single model!")
    else:
        print(f"  Average improvement: {avg_improvement:.1f}%")
        print(f"  Small improvement in this case because:")
        print(f"    - Simple problem (3 features, linear relationships)")
        print(f"    - All models converge to similar solutions")
        print(f"    - Low model diversity (same algorithm, same data)")
    
    print("\nConclusion:")
    print(f"  - Training {ENSEMBLE_SIZE} models gives {best_speedup:.2f}x speedup via parallelization")
    if avg_improvement > 1.0:
        print(f"  - Averaging all {ENSEMBLE_SIZE} models improves accuracy by {avg_improvement:.1f}%")
    else:
        print(f"  - Ensemble averaging demonstrates the concept (improvement varies)")
    print(f"  - Best of both worlds: faster training + ensemble capability!")
    
    print("\n" + "="*70)
    print("WHEN ENSEMBLE AVERAGING WORKS BEST")
    print("="*70)
    print("""
Large improvement (5-20%) when:
  ✓ Complex problems (many features, non-linear)
  ✓ Diverse models (different algorithms or architectures)
  ✓ Different training data (bagging, bootstrapping)
  ✓ High variance models (prone to overfitting)
  ✓ Noisy data (averaging reduces noise)

Small improvement (<2%) when:
  • Simple problems (few features, linear)
  • Homogeneous models (same algorithm, same data)
  • Models already converged well
  • Low-noise data
  
Our case: Simple problem with homogeneous models
Real-world: Often see 5-10% improvement with diverse ensembles
""")
    
    print("\n" + "="*70)
    print("REAL-WORLD APPLICATIONS")
    print("="*70)
    print("""
Ensemble averaging is used in:
  - Random Forests: Average hundreds of decision trees
  - Kaggle Competitions: Winners often ensemble 50-100 models
  - Production ML: Reduce prediction variance and improve robustness
  - Weather Forecasting: Average multiple physics models
  - Medical Diagnosis: Combine multiple diagnostic models for safety
  
Your setup is perfect for ensemble averaging:
  - 8 diverse models (different random initializations)
  - Independent training (ideal for parallelization)
  - Natural diversity leads to error cancellation
    """)
    
    print("="*70)
    print("ENSEMBLE STRATEGIES COMPARISON")
    print("="*70)
    print(f"""
1. Select Best (current in quick_test_clean.py):
   - Uses only 1 model (wastes other {ENSEMBLE_SIZE-1})
   - Fast inference (1 prediction)
   - Lower accuracy
   
2. Simple Average (this version):
   - Uses ALL {ENSEMBLE_SIZE} models
   - {ENSEMBLE_SIZE}x slower inference ({ENSEMBLE_SIZE} predictions)
   - ~{avg_improvement:.1f}% better accuracy
   
3. Weighted Average (advanced):
   - Weight better models more
   - Similar speed to simple average
   - Slightly better than simple average
   
4. Model Distillation (advanced):
   - Train new model to mimic ensemble
   - Fast inference (1 prediction)
   - Keeps most of ensemble accuracy
    """)
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE!")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    run_test()