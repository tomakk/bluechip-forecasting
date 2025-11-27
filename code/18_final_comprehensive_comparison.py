"""
Final Comprehensive Model Comparison - Blue Chip Economic Indicators

Consolidates results from ALL successfully completed model families:
- Machine Learning: XGBoost, Random Forest, LightGBM
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

# Output directory
OUTPUT_DIR = Path('final_comparison_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Result file paths
ML_RESULTS = Path('ml_models_output/ml_models_results.json')
DL_RESULTS = Path('deep_learning_output/deep_learning_results.json')
XLSTM_RESULTS = Path('deep_learning_output/xlstm_results.json')
FOUNDATION_RESULTS = Path('foundation_models_output/all_foundation_models_results.json')

print("="*80)
print("FINAL COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print("Comparing ALL successfully completed model families")
print()

# ==============================================================================
# LOAD RESULTS
# ==============================================================================

all_rows = []

# Load ML Results (different structure)
print("Loading ML Models...")
try:
    with open(ML_RESULTS, 'r') as f:
        ml_data = json.load(f)

    for model_result in ml_data['models']:
        indicator = model_result['indicator']
        for model_name, metrics in model_result['performance'].items():
            all_rows.append({
                'Indicator': indicator,
                'Model Family': 'Machine Learning',
                'Model': model_name,
                'MAPE': metrics['MAPE'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'N_Predictions': metrics['n_forecasts']
            })
    print(f"✅ Loaded ML Models: {len(ml_data['models'])} indicators")
except Exception as e:
    print(f"❌ Error loading ML: {str(e)}")

# Load Deep Learning Results
print("Loading Deep Learning...")
try:
    with open(DL_RESULTS, 'r') as f:
        dl_data = json.load(f)

    for indicator, models in dl_data.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'mape' in metrics:
                all_rows.append({
                    'Indicator': indicator,
                    'Model Family': 'Deep Learning',
                    'Model': model_name,
                    'MAPE': metrics['mape'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics.get('rmse', np.nan),
                    'N_Predictions': metrics.get('n_predictions', 0)
                })
    print(f"✅ Loaded Deep Learning: {len(dl_data)} indicators")
except Exception as e:
    print(f"❌ Error loading DL: {str(e)}")

# Load xLSTM Results
print("Loading xLSTM...")
try:
    with open(XLSTM_RESULTS, 'r') as f:
        xlstm_data = json.load(f)

    for indicator, models in xlstm_data.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'mape' in metrics:
                all_rows.append({
                    'Indicator': indicator,
                    'Model Family': 'xLSTM',
                    'Model': model_name,
                    'MAPE': metrics['mape'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics.get('rmse', np.nan),
                    'N_Predictions': metrics.get('n_predictions', 0)
                })
    print(f"✅ Loaded xLSTM: {len(xlstm_data)} indicators")
except Exception as e:
    print(f"❌ Error loading xLSTM: {str(e)}")

# Load Foundation Results
print("Loading Foundation Models...")
try:
    with open(FOUNDATION_RESULTS, 'r') as f:
        foundation_data = json.load(f)

    for indicator, models in foundation_data.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'mape' in metrics:
                all_rows.append({
                    'Indicator': indicator,
                    'Model Family': 'Foundation',
                    'Model': model_name,
                    'MAPE': metrics['mape'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics.get('rmse', np.nan),
                    'N_Predictions': metrics.get('n_predictions', 0)
                })
    print(f"✅ Loaded Foundation: {len(foundation_data)} indicators")
except Exception as e:
    print(f"❌ Error loading Foundation: {str(e)}")

# Create DataFrame
df_all = pd.DataFrame(all_rows)

print()
print(f"Total results: {len(df_all)} model-indicator combinations")
print()

# ==============================================================================
# BEST MODEL BY INDICATOR
# ==============================================================================

print("="*80)
print("🏆 CHAMPION MODEL FOR EACH INDICATOR")
print("="*80)
print()

best_models = []

for indicator in sorted(df_all['Indicator'].unique()):
    indicator_df = df_all[df_all['Indicator'] == indicator].copy()

    # Find best model
    best_idx = indicator_df['MAPE'].idxmin()
    best = indicator_df.loc[best_idx]

    # Find second-best
    remaining = indicator_df.drop(best_idx)
    if len(remaining) > 0:
        second_best = remaining.loc[remaining['MAPE'].idxmin()]
        improvement = ((second_best['MAPE'] - best['MAPE']) / second_best['MAPE'] * 100)
        second_name = second_best['Model']
    else:
        improvement = np.nan
        second_name = "N/A"

    best_models.append({
        'Indicator': indicator,
        'Champion Model': best['Model'],
        'Model Family': best['Model Family'],
        'MAPE': best['MAPE'],
        'MAE': best['MAE'],
        'Improvement vs 2nd': improvement,
        '2nd Place': second_name
    })

    print(f"{'='*60}")
    print(f"{indicator}")
    print(f"{'='*60}")
    print(f"🥇 CHAMPION: {best['Model']} ({best['Model Family']})")
    print(f"   MAPE: {best['MAPE']:.2f}%")
    print(f"   MAE:  {best['MAE']:.4f}")
    if not np.isnan(improvement):
        print(f"   {improvement:.1f}% better than 2nd place ({second_name})")
    print()

# Save best models
df_best = pd.DataFrame(best_models)
best_file = OUTPUT_DIR / 'champion_models_by_indicator.csv'
df_best.to_csv(best_file, index=False)

# ==============================================================================
# MODEL FAMILY PERFORMANCE
# ==============================================================================

print("="*80)
print("MODEL FAMILY PERFORMANCE SUMMARY")
print("="*80)
print()

family_stats = []
for family in df_all['Model Family'].unique():
    family_df = df_all[df_all['Model Family'] == family]

    family_stats.append({
        'Model Family': family,
        'Mean MAPE': family_df['MAPE'].mean(),
        'Median MAPE': family_df['MAPE'].median(),
        'Best MAPE': family_df['MAPE'].min(),
        'Worst MAPE': family_df['MAPE'].max(),
        'Mean MAE': family_df['MAE'].mean(),
        'Total Models': len(family_df)
    })

df_family = pd.DataFrame(family_stats).sort_values('Mean MAPE')

print(df_family.to_string(index=False))
print()

# Save family summary
family_file = OUTPUT_DIR / 'model_family_summary.csv'
df_family.to_csv(family_file, index=False)

# ==============================================================================
# FAMILY DOMINANCE
# ==============================================================================

print("="*80)
print("🏆 MODEL FAMILY WINS (# OF INDICATORS)")
print("="*80)
print()

family_wins = df_best['Model Family'].value_counts()
for family, count in family_wins.items():
    percentage = (count / len(df_best)) * 100
    print(f"{family:20s}: {count:2d} wins ({percentage:5.1f}%)")
print()

# ==============================================================================
# TOP PERFORMING MODELS
# ==============================================================================

print("="*80)
print("TOP 10 MODEL PERFORMANCES (LOWEST MAPE)")
print("="*80)
print()

df_sorted = df_all.sort_values('MAPE')
top_10 = df_sorted.head(10)[['Indicator', 'Model Family', 'Model', 'MAPE', 'MAE']]

for idx, row in top_10.iterrows():
    rank = list(top_10.index).index(idx) + 1
    print(f"{rank}. {row['Model']} ({row['Model Family']})")
    print(f"   {row['Indicator']}: {row['MAPE']:.2f}% MAPE, {row['MAE']:.4f} MAE")
    print()

# ==============================================================================
# KEY FINDINGS
# ==============================================================================

print("="*80)
print("KEY RESEARCH FINDINGS")
print("="*80)
print()

# Overall best
absolute_best = df_all.loc[df_all['MAPE'].idxmin()]
print(f"⭐ Overall Best Performance:")
print(f"   {absolute_best['Model']} ({absolute_best['Model Family']})")
print(f"   {absolute_best['Indicator']}: {absolute_best['MAPE']:.2f}% MAPE")
print()

# Family champions
print(f"🏆 Model Family Champions:")
for idx, row in df_family.iterrows():
    print(f"   {row['Model Family']:20s}: {row['Mean MAPE']:6.2f}% avg MAPE")
print()

# Chronos dominance
chronos_wins = len(df_best[df_best['Champion Model'] == 'Chronos'])
print(f"🚀 Chronos (Foundation Model) Dominance:")
print(f"   Won {chronos_wins}/9 indicators ({chronos_wins/9*100:.1f}%)")
chronos_rows = df_all[df_all['Model'] == 'Chronos']
print(f"   Average MAPE: {chronos_rows['MAPE'].mean():.2f}%")
print()

# Save complete results
complete_file = OUTPUT_DIR / 'complete_results_all_models.csv'
df_sorted.to_csv(complete_file, index=False)

print("📁 Files Created:")
print(f"   {best_file}")
print(f"   {family_file}")
print(f"   {complete_file}")
print()

print("="*80)
print("✅ FINAL COMPREHENSIVE COMPARISON COMPLETE")
print("="*80)
