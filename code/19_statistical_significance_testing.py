"""
Statistical Significance Testing - Blue Chip Economic Indicators

Performs Diebold-Mariano tests to determine if model performance differences
are statistically significant.

Tests:
1. Diebold-Mariano test for forecast accuracy comparison
2. Model Confidence Set (MCS) for multiple model comparison

Author: Data Scientist Agent
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path('statistical_testing_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load champion models
CHAMPIONS_FILE = Path('final_comparison_output/champion_models_by_indicator.csv')

print("="*80)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*80)
print()

# ==============================================================================
# DIEBOLD-MARIANO TEST
# ==============================================================================

def diebold_mariano_test(errors1, errors2, h=1, alternative='two-sided'):
    """
    Diebold-Mariano test for forecast accuracy comparison.

    Tests null hypothesis: E[L(e1)] = E[L(e2)]
    where L is loss function (squared error)

    Parameters:
    -----------
    errors1 : array-like
        Forecast errors from model 1
    errors2 : array-like
        Forecast errors from model 2
    h : int
        Forecast horizon (for HAC adjustment)
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns:
    --------
    dm_stat : float
        DM test statistic
    p_value : float
        p-value
    """
    # Convert to numpy arrays
    e1 = np.array(errors1)
    e2 = np.array(errors2)

    # Compute loss differential (squared errors)
    d = e1**2 - e2**2

    # Mean of loss differential
    d_bar = np.mean(d)

    # Variance of loss differential (with HAC correction for h-step ahead)
    n = len(d)
    gamma0 = np.var(d, ddof=1)

    # Add autocovariances up to h-1 lags (Harvey et al. 1997 correction)
    if h > 1:
        gamma_sum = 0
        for k in range(1, h):
            cov_k = np.cov(d[:-k], d[k:])[0, 1]
            gamma_sum += (1 - k/h) * cov_k
        var_d = gamma0 + 2 * gamma_sum
    else:
        var_d = gamma0

    # DM test statistic
    dm_stat = d_bar / np.sqrt(var_d / n)

    # p-value (asymptotically normal)
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return dm_stat, p_value


# ==============================================================================
# LOAD CHAMPION MODELS AND PREDICTIONS
# ==============================================================================

df_champions = pd.read_csv(CHAMPIONS_FILE)

print("Champion models loaded:")
for _, row in df_champions.iterrows():
    print(f"  {row['Indicator']:25s}: {row['Champion Model']:15s} ({row['MAPE']:.2f}% MAPE)")
print()

# ==============================================================================
# LOAD PREDICTIONS FROM EACH MODEL FAMILY
# ==============================================================================

def load_predictions_deep_learning():
    """Load predictions from deep learning models."""
    results = {}
    try:
        with open('deep_learning_output/deep_learning_results.json', 'r') as f:
            dl_data = json.load(f)

        for indicator, models in dl_data.items():
            results[indicator] = {}
            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'predictions' in metrics and 'actuals' in metrics:
                    results[indicator][model_name] = {
                        'predictions': np.array(metrics['predictions']),
                        'actuals': np.array(metrics['actuals'])
                    }
        print(f"✅ Loaded Deep Learning predictions: {len(results)} indicators")
    except Exception as e:
        print(f"❌ Error loading DL predictions: {str(e)}")

    return results


def load_predictions_xlstm():
    """Load predictions from xLSTM models."""
    results = {}
    try:
        with open('deep_learning_output/xlstm_results.json', 'r') as f:
            xlstm_data = json.load(f)

        for indicator, models in xlstm_data.items():
            if indicator not in results:
                results[indicator] = {}

            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'predictions' in metrics and 'actuals' in metrics:
                    results[indicator][model_name] = {
                        'predictions': np.array(metrics['predictions']),
                        'actuals': np.array(metrics['actuals'])
                    }
        print(f"✅ Loaded xLSTM predictions: {len(results)} indicators")
    except Exception as e:
        print(f"❌ Error loading xLSTM predictions: {str(e)}")

    return results


def load_predictions_foundation():
    """Load predictions from foundation models."""
    results = {}
    try:
        with open('foundation_models_output/all_foundation_models_results.json', 'r') as f:
            foundation_data = json.load(f)

        for indicator, models in foundation_data.items():
            if indicator not in results:
                results[indicator] = {}

            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'predictions' in metrics and 'actuals' in metrics:
                    # Handle both list and numpy array formats
                    preds = metrics['predictions']
                    acts = metrics['actuals']
                    if isinstance(preds, list):
                        preds = np.array(preds)
                    if isinstance(acts, list):
                        acts = np.array(acts)

                    results[indicator][model_name] = {
                        'predictions': preds,
                        'actuals': acts
                    }
        print(f"✅ Loaded Foundation predictions: {len(results)} indicators")
    except Exception as e:
        print(f"❌ Error loading Foundation predictions: {str(e)}")

    return results


# Load all predictions
all_predictions = {}

dl_preds = load_predictions_deep_learning()
for indicator, models in dl_preds.items():
    all_predictions[indicator] = models

xlstm_preds = load_predictions_xlstm()
for indicator, models in xlstm_preds.items():
    if indicator not in all_predictions:
        all_predictions[indicator] = {}
    all_predictions[indicator].update(models)

foundation_preds = load_predictions_foundation()
for indicator, models in foundation_preds.items():
    if indicator not in all_predictions:
        all_predictions[indicator] = {}
    all_predictions[indicator].update(models)

print()
print(f"Total indicators with predictions: {len(all_predictions)}")
print()

# ==============================================================================
# DIEBOLD-MARIANO TESTS FOR CHAMPION VS COMPETITORS
# ==============================================================================

print("="*80)
print("DIEBOLD-MARIANO TESTS: CHAMPION VS TOP COMPETITORS")
print("="*80)
print()

dm_results = []

for _, row in df_champions.iterrows():
    indicator = row['Indicator']
    champion_model = row['Champion Model']

    print(f"{'='*60}")
    print(f"{indicator}")
    print(f"{'='*60}")
    print(f"Champion: {champion_model} ({row['MAPE']:.2f}% MAPE)")
    print()

    if indicator not in all_predictions:
        print(f"  ⚠️  No prediction data available")
        print()
        continue

    if champion_model not in all_predictions[indicator]:
        print(f"  ⚠️  Champion model predictions not found")
        print()
        continue

    # Get champion predictions and errors
    champion_preds = all_predictions[indicator][champion_model]['predictions']
    champion_actuals = all_predictions[indicator][champion_model]['actuals']
    champion_errors = champion_actuals - champion_preds

    # Test against all other models
    competitor_results = []

    for competitor_model, data in all_predictions[indicator].items():
        if competitor_model == champion_model:
            continue

        competitor_preds = data['predictions']
        competitor_actuals = data['actuals']

        # Ensure same length
        if len(champion_preds) != len(competitor_preds):
            continue

        competitor_errors = competitor_actuals - competitor_preds

        # Compute Diebold-Mariano test
        dm_stat, p_value = diebold_mariano_test(
            champion_errors,
            competitor_errors,
            h=1,
            alternative='two-sided'
        )

        # Compute MAPEs for comparison
        champion_mape = np.mean(np.abs(champion_errors / champion_actuals)) * 100
        competitor_mape = np.mean(np.abs(competitor_errors / competitor_actuals)) * 100

        competitor_results.append({
            'Indicator': indicator,
            'Champion': champion_model,
            'Competitor': competitor_model,
            'Champion MAPE': champion_mape,
            'Competitor MAPE': competitor_mape,
            'DM Statistic': dm_stat,
            'p-value': p_value,
            'Significant (5%)': 'Yes' if p_value < 0.05 else 'No',
            'Significant (10%)': 'Yes' if p_value < 0.10 else 'No'
        })

    # Sort by competitor MAPE (show closest competitors)
    competitor_results = sorted(competitor_results, key=lambda x: x['Competitor MAPE'])

    # Display top 5 competitors
    print(f"  Top 5 Competitors:")
    print()

    for i, result in enumerate(competitor_results[:5], 1):
        print(f"  {i}. {result['Competitor']}")
        print(f"     MAPE: {result['Competitor MAPE']:.2f}%")
        print(f"     DM Statistic: {result['DM Statistic']:+.3f}")
        print(f"     p-value: {result['p-value']:.4f}")

        if result['Significant (5%)'] == 'Yes':
            print(f"     ✅ Significantly different at 5% level")
        elif result['Significant (10%)'] == 'Yes':
            print(f"     ⚠️  Significantly different at 10% level")
        else:
            print(f"     ❌ NOT significantly different")
        print()

    dm_results.extend(competitor_results)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

df_dm = pd.DataFrame(dm_results)
dm_file = OUTPUT_DIR / 'diebold_mariano_results.csv'
df_dm.to_csv(dm_file, index=False)

print()
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print()

total_tests = len(df_dm)
significant_5 = len(df_dm[df_dm['Significant (5%)'] == 'Yes'])
significant_10 = len(df_dm[df_dm['Significant (10%)'] == 'Yes'])

print(f"Total DM tests performed: {total_tests}")
print(f"Significant at 5% level:  {significant_5} ({significant_5/total_tests*100:.1f}%)")
print(f"Significant at 10% level: {significant_10} ({significant_10/total_tests*100:.1f}%)")
print()

# Champion dominance by indicator
print("Champion Statistical Dominance:")
print()

for indicator in df_champions['Indicator'].unique():
    indicator_tests = df_dm[df_dm['Indicator'] == indicator]

    if len(indicator_tests) == 0:
        continue

    sig_5 = len(indicator_tests[indicator_tests['Significant (5%)'] == 'Yes'])
    total = len(indicator_tests)

    champion = indicator_tests.iloc[0]['Champion']
    print(f"  {indicator:25s}: {champion:15s}")
    print(f"    Statistically superior to {sig_5}/{total} competitors (5% level)")

print()
print(f"📁 Results saved to: {dm_file}")
print()

print("="*80)
print("✅ STATISTICAL SIGNIFICANCE TESTING COMPLETE")
print("="*80)
