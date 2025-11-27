"""
Publication-Ready Figures - Blue Chip Economic Indicators
Addressing Peer Review Figure Requirements

Creates high-quality figures (300 DPI) for academic publication:
1. Performance heatmap (all models × all indicators)
2. Champion models distribution
3. Model family comparison with confidence intervals
4. DM test significance matrix
5. Feature engineering ablation comparison
6. Foundation model outlier analysis
7. Sample size and power analysis
8. Model family win rates
9. Top performers bar chart
10. Horizon comparison (if applicable)

Author: Scientific Visualization Specialist
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set publication quality defaults
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Output directory
OUTPUT_DIR = Path('publication_figures_v2')
OUTPUT_DIR.mkdir(exist_ok=True)

# Data paths
RESULTS_PATH = Path('final_comparison_output/complete_results_all_models.csv')
CHAMPIONS_PATH = Path('final_comparison_output/champion_models_by_indicator.csv')
DM_RESULTS_PATH = Path('peer_review_response/complete_dm_results.csv')
CI_RESULTS_PATH = Path('peer_review_response/bootstrap_confidence_intervals.csv')
ABLATION_PATH = Path('peer_review_response/feature_engineering_ablation.csv')

print("="*80)
print("CREATING PUBLICATION-READY FIGURES")
print("="*80)
print()

# Load all data
df_results = pd.read_csv(RESULTS_PATH)
df_champions = pd.read_csv(CHAMPIONS_PATH)
df_dm = pd.read_csv(DM_RESULTS_PATH)
df_ci = pd.read_csv(CI_RESULTS_PATH)
df_ablation = pd.read_csv(ABLATION_PATH)

print(f"✅ Loaded {len(df_results)} results")
print(f"✅ Loaded {len(df_champions)} champions")
print(f"✅ Loaded {len(df_dm)} DM test results")
print(f"✅ Loaded {len(df_ci)} confidence intervals")
print(f"✅ Loaded {len(df_ablation)} ablation results")
print()

# Color palettes
FAMILY_COLORS = {
    'Machine Learning': '#e74c3c',
    'Deep Learning': '#3498db',
    'xLSTM': '#9b59b6',
    'Foundation': '#2ecc71'
}

MODEL_COLORS = {
    'XGBoost': '#e74c3c',
    'RandomForest': '#c0392b',
    'LightGBM': '#f39c12',
    'LSTM': '#3498db',
    'GRU': '#2980b9',
    'Transformer': '#1abc9c',
    'sLSTM': '#9b59b6',
    'mLSTM': '#8e44ad',
    'Chronos': '#2ecc71',
    'Naive-Seasonal': '#95a5a6'
}

# ==============================================================================
# FIGURE 1: PERFORMANCE HEATMAP
# ==============================================================================

print("Creating Figure 1: Performance Heatmap...")

# Pivot data for heatmap
df_pivot = df_results.pivot_table(
    values='MAPE',
    index='Model',
    columns='Indicator',
    aggfunc='first'
)

# Reorder models by family
model_order = ['XGBoost', 'RandomForest', 'LightGBM', 'LSTM', 'GRU', 'Transformer',
               'sLSTM', 'mLSTM', 'Chronos', 'Naive-Seasonal']
model_order = [m for m in model_order if m in df_pivot.index]
df_pivot = df_pivot.reindex(model_order)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Custom colormap (green = good, red = bad)
cmap = LinearSegmentedColormap.from_list('mape', ['#27ae60', '#f1c40f', '#e74c3c'])

# Cap values for visualization
df_plot = df_pivot.clip(upper=100)

# Heatmap
sns.heatmap(df_plot, annot=True, fmt='.1f', cmap=cmap,
            ax=ax, cbar_kws={'label': 'MAPE (%)'}, vmin=0, vmax=50,
            linewidths=0.5, linecolor='white')

ax.set_xlabel('Economic Indicator', fontweight='bold')
ax.set_ylabel('Model', fontweight='bold')
ax.set_title('Forecast Accuracy Heatmap: MAPE (%) Across Models and Indicators\n(Lower is Better)',
             fontweight='bold', pad=20)

# Rotate x labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure1_performance_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure1_performance_heatmap.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 1 saved")

# ==============================================================================
# FIGURE 2: CHAMPION MODELS BY INDICATOR
# ==============================================================================

print("Creating Figure 2: Champion Models Distribution...")

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by MAPE
df_champ_sorted = df_champions.sort_values('MAPE')

# Create bars with family colors
colors = [FAMILY_COLORS.get(row['Model Family'], '#95a5a6') for _, row in df_champ_sorted.iterrows()]

bars = ax.barh(df_champ_sorted['Indicator'], df_champ_sorted['MAPE'], color=colors, edgecolor='white', height=0.7)

# Add model names as text
for i, (bar, (_, row)) in enumerate(zip(bars, df_champ_sorted.iterrows())):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{row['Champion Model']} ({row['MAPE']:.1f}%)",
            va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('MAPE (%)', fontweight='bold')
ax.set_ylabel('Economic Indicator', fontweight='bold')
ax.set_title('Champion Model Performance by Indicator', fontweight='bold', pad=20)

# Legend
legend_patches = [Patch(color=color, label=family) for family, color in FAMILY_COLORS.items()]
ax.legend(handles=legend_patches, loc='lower right', title='Model Family')

ax.set_xlim(0, max(df_champ_sorted['MAPE']) * 1.3)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure2_champion_models.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure2_champion_models.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 2 saved")

# ==============================================================================
# FIGURE 3: MODEL FAMILY COMPARISON WITH CONFIDENCE INTERVALS
# ==============================================================================

print("Creating Figure 3: Model Family Comparison with CIs...")

fig, ax = plt.subplots(figsize=(10, 6))

# Group CI results by model and compute family
model_to_family = {
    'XGBoost': 'Machine Learning', 'RandomForest': 'Machine Learning', 'LightGBM': 'Machine Learning',
    'LSTM': 'Deep Learning', 'GRU': 'Deep Learning', 'Transformer': 'Deep Learning',
    'sLSTM': 'xLSTM', 'mLSTM': 'xLSTM',
    'Chronos': 'Foundation', 'Naive-Seasonal': 'Foundation'
}

df_ci['Model Family'] = df_ci['Model'].map(model_to_family)

# Aggregate by family
family_stats = df_ci.groupby('Model Family').agg({
    'MAPE': ['mean', 'std', 'median'],
    'CI_Lower_95': 'mean',
    'CI_Upper_95': 'mean',
    'Model': 'count'
}).round(2)
family_stats.columns = ['Mean_MAPE', 'Std_MAPE', 'Median_MAPE', 'Avg_CI_Lower', 'Avg_CI_Upper', 'N_Results']
family_stats = family_stats.reset_index()
family_stats = family_stats.sort_values('Mean_MAPE')

# Plot
x_pos = np.arange(len(family_stats))
colors = [FAMILY_COLORS.get(f, '#95a5a6') for f in family_stats['Model Family']]

# Mean with error bars (use std as error)
bars = ax.bar(x_pos, family_stats['Mean_MAPE'],
              yerr=family_stats['Std_MAPE'], capsize=5,
              color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add median line
for i, (_, row) in enumerate(family_stats.iterrows()):
    ax.plot([i-0.35, i+0.35], [row['Median_MAPE'], row['Median_MAPE']],
            'k--', linewidth=2, label='Median' if i == 0 else None)

ax.set_xticks(x_pos)
ax.set_xticklabels(family_stats['Model Family'], rotation=0)
ax.set_xlabel('Model Family', fontweight='bold')
ax.set_ylabel('Mean MAPE (%) ± Std Dev', fontweight='bold')
ax.set_title('Model Family Performance Comparison\n(Error bars show standard deviation, dashed line shows median)',
             fontweight='bold', pad=20)

ax.grid(True, axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure3_family_comparison_ci.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure3_family_comparison_ci.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 3 saved")

# ==============================================================================
# FIGURE 4: DIEBOLD-MARIANO TEST SIGNIFICANCE MATRIX
# ==============================================================================

print("Creating Figure 4: DM Test Significance Matrix...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create significance matrix
indicators = df_dm['Indicator'].unique()
competitors = df_dm['Competitor'].unique()

# Create pivot table for p-values
df_dm_pivot = df_dm.pivot_table(
    values='p_value',
    index='Indicator',
    columns='Competitor',
    aggfunc='first'
)

# Convert to significance levels
def significance_level(p):
    if pd.isna(p):
        return 0
    elif p < 0.01:
        return 3  # *** highly significant
    elif p < 0.05:
        return 2  # ** significant
    elif p < 0.10:
        return 1  # * marginally significant
    else:
        return 0  # not significant

df_sig = df_dm_pivot.apply(lambda x: x.apply(significance_level))

# Custom colormap
colors_sig = ['#f0f0f0', '#f1c40f', '#e67e22', '#c0392b']
cmap_sig = LinearSegmentedColormap.from_list('sig', colors_sig)

# Plot heatmap
im = ax.imshow(df_sig.values, cmap=cmap_sig, aspect='auto', vmin=0, vmax=3)

# Add text annotations
for i in range(len(df_sig.index)):
    for j in range(len(df_sig.columns)):
        val = df_sig.iloc[i, j]
        p = df_dm_pivot.iloc[i, j]
        if pd.notna(p):
            text = '*' * int(val) if val > 0 else 'ns'
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='white' if val >= 2 else 'black')

ax.set_xticks(np.arange(len(df_sig.columns)))
ax.set_yticks(np.arange(len(df_sig.index)))
ax.set_xticklabels(df_sig.columns, rotation=45, ha='right')
ax.set_yticklabels(df_sig.index)

ax.set_xlabel('Competitor Model', fontweight='bold')
ax.set_ylabel('Indicator', fontweight='bold')
ax.set_title('Diebold-Mariano Test Results: Champion vs Competitors\n(*** p<0.01, ** p<0.05, * p<0.10, ns = not significant)',
             fontweight='bold', pad=20)

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['ns', '*', '**', '***'])
cbar.set_label('Significance Level')

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure4_dm_significance_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure4_dm_significance_matrix.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 4 saved")

# ==============================================================================
# FIGURE 5: FEATURE ENGINEERING ABLATION STUDY
# ==============================================================================

print("Creating Figure 5: Feature Engineering Ablation Study...")

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by improvement
df_abl_sorted = df_ablation.sort_values('Improvement_pct', ascending=True)

x = np.arange(len(df_abl_sorted))
width = 0.35

# Bars for raw and engineered
bars1 = ax.barh(x - width/2, df_abl_sorted['Raw_Features_MAPE'], width,
                label='Raw Features (3)', color='#e74c3c', alpha=0.8)
bars2 = ax.barh(x + width/2, df_abl_sorted['Engineered_Features_MAPE'], width,
                label='Engineered Features (21)', color='#27ae60', alpha=0.8)

# Add improvement annotations
for i, (_, row) in enumerate(df_abl_sorted.iterrows()):
    ax.annotate(f"+{row['Improvement_pct']:.0f}%",
                xy=(max(row['Raw_Features_MAPE'], row['Engineered_Features_MAPE']) + 1, i),
                va='center', fontsize=9, fontweight='bold', color='#27ae60')

ax.set_yticks(x)
ax.set_yticklabels(df_abl_sorted['Indicator'])
ax.set_xlabel('MAPE (%)', fontweight='bold')
ax.set_ylabel('Economic Indicator', fontweight='bold')
ax.set_title('Feature Engineering Impact on XGBoost Performance\n(Green percentages show improvement from feature engineering)',
             fontweight='bold', pad=20)

ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure5_ablation_study.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure5_ablation_study.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 5 saved")

# ==============================================================================
# FIGURE 6: FOUNDATION MODEL PERFORMANCE DISTRIBUTION
# ==============================================================================

print("Creating Figure 6: Foundation Model Performance Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get Chronos results
chronos_results = df_results[df_results['Model'] == 'Chronos'].sort_values('MAPE')

# Left plot: All Chronos results
colors_chronos = ['#27ae60' if m < 20 else '#f1c40f' if m < 50 else '#e74c3c'
                  for m in chronos_results['MAPE']]
ax1.barh(chronos_results['Indicator'], chronos_results['MAPE'], color=colors_chronos, edgecolor='white')

ax1.axvline(x=20, color='#27ae60', linestyle='--', linewidth=2, label='Good (<20%)')
ax1.axvline(x=50, color='#e74c3c', linestyle='--', linewidth=2, label='Outlier threshold (50%)')

ax1.set_xlabel('MAPE (%)', fontweight='bold')
ax1.set_ylabel('Economic Indicator', fontweight='bold')
ax1.set_title('Chronos Performance by Indicator', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, axis='x', alpha=0.3)

# Right plot: Summary statistics comparison
stats_labels = ['Mean\n(All)', 'Median\n(All)', 'Mean\n(Non-Outliers)', 'Trimmed Mean\n(10%)']
stats_values = [
    chronos_results['MAPE'].mean(),
    chronos_results['MAPE'].median(),
    chronos_results[chronos_results['MAPE'] <= 50]['MAPE'].mean(),
    np.mean(chronos_results['MAPE'].sort_values().iloc[1:-1])  # Simple trim
]

colors_stats = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6']
ax2.bar(stats_labels, stats_values, color=colors_stats, edgecolor='black', linewidth=1.5)

for i, v in enumerate(stats_values):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

ax2.set_ylabel('MAPE (%)', fontweight='bold')
ax2.set_title('Summary Statistics Comparison\n(Outlier Treatment Options)', fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure6_foundation_model_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure6_foundation_model_analysis.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 6 saved")

# ==============================================================================
# FIGURE 7: MODEL FAMILY WIN RATES
# ==============================================================================

print("Creating Figure 7: Model Family Win Rates...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart of wins
family_wins = df_champions['Model Family'].value_counts()
colors_pie = [FAMILY_COLORS.get(f, '#95a5a6') for f in family_wins.index]

wedges, texts, autotexts = ax1.pie(family_wins.values, labels=family_wins.index,
                                    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(family_wins.values))})',
                                    colors=colors_pie, startangle=90,
                                    explode=[0.05 if f == family_wins.index[0] else 0 for f in family_wins.index])

ax1.set_title('Model Family Win Distribution\n(% of Champion Models)', fontweight='bold')

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# Right: Stacked bar showing model distribution within family
model_wins = df_champions['Champion Model'].value_counts()

# Group by family
family_model_counts = {}
for model, count in model_wins.items():
    family = model_to_family.get(model, 'Unknown')
    if family not in family_model_counts:
        family_model_counts[family] = {}
    family_model_counts[family][model] = count

# Create stacked bar
families = list(family_model_counts.keys())
bottom = np.zeros(len(families))

for model in model_wins.index:
    counts = []
    for family in families:
        counts.append(family_model_counts.get(family, {}).get(model, 0))

    ax2.bar(families, counts, bottom=bottom, label=model,
            color=MODEL_COLORS.get(model, '#95a5a6'), edgecolor='white')
    bottom += counts

ax2.set_ylabel('Number of Wins', fontweight='bold')
ax2.set_xlabel('Model Family', fontweight='bold')
ax2.set_title('Champion Models within Each Family', fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure7_win_rates.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure7_win_rates.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 7 saved")

# ==============================================================================
# FIGURE 8: TOP 10 MODEL PERFORMANCES
# ==============================================================================

print("Creating Figure 8: Top 10 Model Performances...")

fig, ax = plt.subplots(figsize=(12, 8))

# Get top 10 performers
top10 = df_results.nsmallest(10, 'MAPE')

# Create labels
top10['Label'] = top10['Model'] + ' (' + top10['Indicator'].str[:15] + ')'

# Colors by family
colors_top = [FAMILY_COLORS.get(row['Model Family'], '#95a5a6') for _, row in top10.iterrows()]

ax.barh(range(len(top10)), top10['MAPE'], color=colors_top, edgecolor='white', height=0.7)

# Add MAPE values
for i, (_, row) in enumerate(top10.iterrows()):
    ax.text(row['MAPE'] + 0.2, i, f"{row['MAPE']:.2f}%", va='center', fontweight='bold')

ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['Label'])
ax.set_xlabel('MAPE (%)', fontweight='bold')
ax.set_title('Top 10 Best Performing Model-Indicator Combinations', fontweight='bold', pad=20)

# Legend
legend_patches = [Patch(color=color, label=family) for family, color in FAMILY_COLORS.items()]
ax.legend(handles=legend_patches, loc='lower right', title='Model Family')

ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()  # Best at top

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure8_top10_performances.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure8_top10_performances.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 8 saved")

# ==============================================================================
# FIGURE 9: CONFIDENCE INTERVAL COMPARISON BY MODEL
# ==============================================================================

print("Creating Figure 9: Confidence Intervals by Model...")

fig, ax = plt.subplots(figsize=(14, 8))

# Get mean CIs per model
model_ci = df_ci.groupby('Model').agg({
    'MAPE': 'mean',
    'CI_Lower_95': 'mean',
    'CI_Upper_95': 'mean'
}).reset_index()
model_ci = model_ci.sort_values('MAPE')

# Plot
colors_ci = [MODEL_COLORS.get(m, '#95a5a6') for m in model_ci['Model']]
x_pos = np.arange(len(model_ci))

# Bars with error bars
ax.bar(x_pos, model_ci['MAPE'], color=colors_ci, edgecolor='black', linewidth=1.5, alpha=0.8)

# Error bars (CI as error)
errors_lower = model_ci['MAPE'] - model_ci['CI_Lower_95']
errors_upper = model_ci['CI_Upper_95'] - model_ci['MAPE']
ax.errorbar(x_pos, model_ci['MAPE'], yerr=[errors_lower, errors_upper],
            fmt='none', color='black', capsize=5, capthick=2, linewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(model_ci['Model'], rotation=45, ha='right')
ax.set_ylabel('Mean MAPE (%)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.set_title('Model Performance with 95% Bootstrap Confidence Intervals\n(Averaged Across All Indicators)',
             fontweight='bold', pad=20)

ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure9_model_confidence_intervals.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure9_model_confidence_intervals.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 9 saved")

# ==============================================================================
# FIGURE 10: INDICATOR DIFFICULTY RANKING
# ==============================================================================

print("Creating Figure 10: Indicator Difficulty Ranking...")

fig, ax = plt.subplots(figsize=(12, 7))

# Calculate difficulty as average MAPE across all models
difficulty = df_results.groupby('Indicator')['MAPE'].agg(['mean', 'std', 'min', 'max']).reset_index()
difficulty.columns = ['Indicator', 'Mean_MAPE', 'Std_MAPE', 'Best_MAPE', 'Worst_MAPE']
difficulty = difficulty.sort_values('Mean_MAPE', ascending=True)

# Color by difficulty
colors_diff = ['#27ae60' if m < 15 else '#f1c40f' if m < 30 else '#e74c3c'
               for m in difficulty['Mean_MAPE']]

ax.barh(difficulty['Indicator'], difficulty['Mean_MAPE'],
        xerr=difficulty['Std_MAPE'], capsize=5,
        color=colors_diff, edgecolor='white', height=0.7, alpha=0.8)

# Add best model MAPE
for i, (_, row) in enumerate(difficulty.iterrows()):
    ax.text(row['Mean_MAPE'] + row['Std_MAPE'] + 2, i,
            f"Best: {row['Best_MAPE']:.1f}%", va='center', fontsize=9)

ax.set_xlabel('Mean MAPE (%) ± Std Dev Across All Models', fontweight='bold')
ax.set_ylabel('Economic Indicator', fontweight='bold')
ax.set_title('Indicator Forecasting Difficulty\n(Green = Easy, Yellow = Moderate, Red = Difficult)',
             fontweight='bold', pad=20)

ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'figure10_indicator_difficulty.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUTPUT_DIR / 'figure10_indicator_difficulty.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print("  ✅ Figure 10 saved")

# ==============================================================================
# SUMMARY
# ==============================================================================

print()
print("="*80)
print("PUBLICATION FIGURES COMPLETE")
print("="*80)
print()

print(f"📁 Output directory: {OUTPUT_DIR}")
print()
print("Figures created:")
print("  1. Performance Heatmap (all models × all indicators)")
print("  2. Champion Models by Indicator")
print("  3. Model Family Comparison with CIs")
print("  4. DM Test Significance Matrix")
print("  5. Feature Engineering Ablation Study")
print("  6. Foundation Model Performance Distribution")
print("  7. Model Family Win Rates")
print("  8. Top 10 Model Performances")
print("  9. Model Confidence Intervals")
print("  10. Indicator Difficulty Ranking")
print()
print("All figures saved in both PNG (300 DPI) and PDF formats.")
print()
print("="*80)
print("✅ ALL PUBLICATION FIGURES GENERATED")
print("="*80)
