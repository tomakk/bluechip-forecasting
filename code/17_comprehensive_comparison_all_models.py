"""
Comprehensive Model Comparison - Blue Chip Economic Indicators

Creates final comparison across ALL model families:
- Classical: ARIMA, SARIMA, ETS, VAR
- ML: XGBoost, Random Forest, LightGBM
- Deep Learning: LSTM, GRU, Transformer
- xLSTM: sLSTM, mLSTM
- Foundation: Chronos

Author: Data Scientist Agent
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('comprehensive_comparison_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Result file paths
CLASSICAL_RESULTS = Path('classical_models_output/classical_models_results_h1.json')
ML_RESULTS = Path('ml_models_output/ml_models_results.json')
DL_RESULTS = Path('deep_learning_output/deep_learning_results.json')
XLSTM_RESULTS = Path('deep_learning_output/xlstm_results.json')
FOUNDATION_RESULTS = Path('foundation_models_output/all_foundation_models_results.json')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print("Comparing ALL model families across 9 economic indicators")
print()

# ==============================================================================
# LOAD ALL RESULTS
# ==============================================================================

def load_results(file_path, model_family):
    """Load results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {model_family}: {len(data)} indicators")
        return data
    except FileNotFoundError:
        print(f"⚠️  {model_family} results not found: {file_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading {model_family}: {str(e)}")
        return {}


# Load all model families
classical_results = load_results(CLASSICAL_RESULTS, "Classical Models")
ml_results = load_results(ML_RESULTS, "ML Models")
dl_results = load_results(DL_RESULTS, "Deep Learning")
xlstm_results = load_results(XLSTM_RESULTS, "xLSTM")
foundation_results = load_results(FOUNDATION_RESULTS, "Foundation Models")

print()

# ==============================================================================
# CONSOLIDATE RESULTS
# ==============================================================================

def extract_metrics(results_dict, model_family):
    """Extract MAPE and MAE metrics from results."""
    rows = []

    for indicator, models in results_dict.items():
        for model_name, metrics in models.items():
            # Skip predictions/actuals fields, only get summary metrics
            if isinstance(metrics, dict) and 'mape' in metrics:
                rows.append({
                    'Indicator': indicator,
                    'Model Family': model_family,
                    'Model': model_name,
                    'MAPE': metrics['mape'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics.get('rmse', np.nan),
                    'N_Predictions': metrics.get('n_predictions', 0)
                })

    return rows


# Extract metrics from all model families
all_rows = []

if classical_results:
    all_rows.extend(extract_metrics(classical_results, 'Classical'))

if ml_results:
    all_rows.extend(extract_metrics(ml_results, 'Machine Learning'))

if dl_results:
    all_rows.extend(extract_metrics(dl_results, 'Deep Learning'))

if xlstm_results:
    all_rows.extend(extract_metrics(xlstm_results, 'xLSTM'))

if foundation_results:
    all_rows.extend(extract_metrics(foundation_results, 'Foundation'))

# Create comprehensive DataFrame
df_all = pd.DataFrame(all_rows)

print(f"Total results: {len(df_all)} model-indicator combinations")
print()

# ==============================================================================
# BEST MODEL BY INDICATOR
# ==============================================================================

print("="*80)
print("BEST MODEL FOR EACH INDICATOR (ACROSS ALL FAMILIES)")
print("="*80)
print()

best_models = []

for indicator in df_all['Indicator'].unique():
    indicator_df = df_all[df_all['Indicator'] == indicator].copy()

    # Find model with lowest MAPE
    best_idx = indicator_df['MAPE'].idxmin()
    best = indicator_df.loc[best_idx]

    # Find second-best for comparison
    remaining = indicator_df.drop(best_idx)
    if len(remaining) > 0:
        second_best = remaining.loc[remaining['MAPE'].idxmin()]
        improvement = ((second_best['MAPE'] - best['MAPE']) / second_best['MAPE'] * 100)
    else:
        improvement = np.nan

    best_models.append({
        'Indicator': indicator,
        'Best Model': best['Model'],
        'Model Family': best['Model Family'],
        'MAPE': best['MAPE'],
        'MAE': best['MAE'],
        'Improvement vs 2nd': improvement
    })

    print(f"{indicator}:")
    print(f"  🏆 CHAMPION: {best['Model']} ({best['Model Family']})")
    print(f"     MAPE: {best['MAPE']:.2f}%")
    print(f"     MAE:  {best['MAE']:.4f}")
    if not np.isnan(improvement):
        print(f"     {improvement:.1f}% better than 2nd place")
    print()

# Save best models summary
df_best = pd.DataFrame(best_models)
best_file = OUTPUT_DIR / 'best_models_by_indicator.csv'
df_best.to_csv(best_file, index=False)

# ==============================================================================
# MODEL FAMILY PERFORMANCE
# ==============================================================================

print("="*80)
print("MODEL FAMILY PERFORMANCE SUMMARY")
print("="*80)
print()

family_summary = df_all.groupby('Model Family').agg({
    'MAPE': ['mean', 'median', 'min', 'max'],
    'MAE': ['mean', 'median'],
    'Indicator': 'count'
}).round(4)

family_summary.columns = ['_'.join(col).strip() for col in family_summary.columns.values]
family_summary = family_summary.rename(columns={'Indicator_count': 'N_Results'})
family_summary = family_summary.sort_values('MAPE_mean')

print(family_summary.to_string())
print()

# Save family summary
family_file = OUTPUT_DIR / 'model_family_summary.csv'
family_summary.to_csv(family_file)

# ==============================================================================
# FAMILY DOMINANCE COUNT
# ==============================================================================

print("="*80)
print("MODEL FAMILY WINS (# OF INDICATORS WHERE FAMILY IS BEST)")
print("="*80)
print()

family_wins = df_best['Model Family'].value_counts()
print(family_wins.to_string())
print()

# ==============================================================================
# FULL RESULTS TABLE
# ==============================================================================

# Sort by indicator and MAPE
df_all_sorted = df_all.sort_values(['Indicator', 'MAPE'])

# Save comprehensive results
full_file = OUTPUT_DIR / 'comprehensive_results_all_models.csv'
df_all_sorted.to_csv(full_file, index=False)

print(f"✅ Comprehensive results saved to: {full_file}")
print()

# ==============================================================================
# TOP MODELS OVERALL
# ==============================================================================

print("="*80)
print("TOP 20 MODEL PERFORMANCES (LOWEST MAPE)")
print("="*80)
print()

top_20 = df_all_sorted.nsmallest(20, 'MAPE')[
    ['Indicator', 'Model Family', 'Model', 'MAPE', 'MAE']
]

print(top_20.to_string(index=False))
print()

# ==============================================================================
# CHRONOS VS ALL COMPETITORS
# ==============================================================================

print("="*80)
print("CHRONOS (FOUNDATION) VS BEST CLASSICAL/ML/DL MODELS")
print("="*80)
print()

chronos_comparison = []

for indicator in df_all['Indicator'].unique():
    indicator_df = df_all[df_all['Indicator'] == indicator].copy()

    # Get Chronos result
    chronos = indicator_df[indicator_df['Model'] == 'Chronos']
    if len(chronos) == 0:
        continue
    chronos_mape = chronos.iloc[0]['MAPE']

    # Get best non-foundation model
    non_foundation = indicator_df[indicator_df['Model Family'] != 'Foundation']
    if len(non_foundation) == 0:
        continue

    best_non_foundation = non_foundation.loc[non_foundation['MAPE'].idxmin()]

    improvement = ((best_non_foundation['MAPE'] - chronos_mape) / best_non_foundation['MAPE'] * 100)

    chronos_comparison.append({
        'Indicator': indicator,
        'Chronos MAPE': chronos_mape,
        'Best Non-Foundation': best_non_foundation['Model'],
        'Best Non-Foundation MAPE': best_non_foundation['MAPE'],
        'Chronos Improvement': improvement
    })

df_chronos = pd.DataFrame(chronos_comparison)
print(df_chronos.to_string(index=False))
print()

# Calculate average improvement
avg_improvement = df_chronos['Chronos Improvement'].mean()
chronos_wins = (df_chronos['Chronos Improvement'] > 0).sum()
total_comparisons = len(df_chronos)

print(f"Chronos wins: {chronos_wins}/{total_comparisons} indicators")
print(f"Average improvement: {avg_improvement:.1f}%")
print()

# Save Chronos comparison
chronos_file = OUTPUT_DIR / 'chronos_vs_competitors.csv'
df_chronos.to_csv(chronos_file, index=False)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("="*80)
print("FINAL SUMMARY")
print("="*80)
print()

print(f"📊 Total Models Tested: {df_all['Model'].nunique()}")
print(f"📈 Total Indicators: {df_all['Indicator'].nunique()}")
print(f"🔬 Total Evaluations: {len(df_all)}")
print()

print("🏆 Model Family Champions:")
for family, count in family_wins.items():
    print(f"   {family}: {count} wins")
print()

print(f"⭐ Best Overall Performance:")
absolute_best = df_all.loc[df_all['MAPE'].idxmin()]
print(f"   {absolute_best['Model']} ({absolute_best['Model Family']})")
print(f"   {absolute_best['Indicator']}: {absolute_best['MAPE']:.2f}% MAPE")
print()

print("📁 Files Created:")
print(f"   {best_file}")
print(f"   {family_file}")
print(f"   {full_file}")
print(f"   {chronos_file}")
print()

print("="*80)
print("COMPREHENSIVE COMPARISON COMPLETE")
print("="*80)
