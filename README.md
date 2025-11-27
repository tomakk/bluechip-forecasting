# Machine Learning vs Foundation Models for Economic Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code, figures, and supplementary materials for the paper:

> **"Machine Learning vs Foundation Models for Economic Forecasting: Evidence from Blue Chip Indicators"**
>
> Submitted to: International Journal of Forecasting

### Abstract

We present the first comprehensive comparison of foundation models, traditional machine learning, and deep learning approaches for forecasting U.S. macroeconomic indicators. Using 49 years of Blue Chip Economic Indicators consensus forecasts (1976-2025), we evaluate 10 models across four families on nine core economic indicators.

**Key Findings:**
- Traditional machine learning (XGBoost) achieves a **44.4% win rate**, outperforming foundation models (33.3%) and deep learning (22.2%)
- XGBoost achieves **1.87% MAPE** on industrial production (98% forecast accuracy)
- Extended LSTM architectures (xLSTM) completely fail on limited economic data (0% win rate)
- **52.8%** of model comparisons show statistically significant differences (Diebold-Mariano tests)

## Repository Structure

```
├── code/                           # Python scripts for analysis
│   ├── 17_comprehensive_comparison_all_models.py
│   ├── 18_final_comprehensive_comparison.py
│   ├── 19_statistical_significance_testing.py
│   ├── 30_address_peer_review_m1_m5.py
│   └── 31_create_publication_figures_v2.py
│
├── figures/                        # Publication-quality figures (300 DPI)
│   ├── figure1_performance_heatmap.png/.pdf
│   ├── figure2_champion_models.png/.pdf
│   ├── figure3_family_comparison_ci.png/.pdf
│   ├── figure4_dm_significance_matrix.png/.pdf
│   ├── figure5_ablation_study.png/.pdf
│   ├── figure6_foundation_model_analysis.png/.pdf
│   ├── figure7_win_rates.png/.pdf
│   ├── figure8_top10_performances.png/.pdf
│   ├── figure9_model_confidence_intervals.png/.pdf
│   └── figure10_indicator_difficulty.png/.pdf
│
├── results/                        # Statistical analysis results
│   ├── complete_dm_results.csv     # Diebold-Mariano test results (all pairs)
│   ├── bootstrap_confidence_intervals.csv
│   ├── chronos_outlier_analysis.json
│   ├── feature_engineering_ablation.csv
│   └── diebold_mariano_results.csv
│
└── supplementary/                  # Supplementary tables
    ├── supplementary_table_s1_complete_rankings.csv
    ├── supplementary_table_s2_hyperparameters.csv
    ├── supplementary_table_s3_performance_by_indicator.csv
    ├── supplementary_table_s4_performance_by_horizon.csv
    ├── supplementary_table_s5_family_statistics.csv
    └── supplementary_table_s6_champion_models.csv
```

## Data Availability

The Blue Chip Economic Indicators dataset is proprietary and cannot be redistributed. Access can be obtained from:
- **Wolters Kluwer**: https://www.wolterskluwer.com/en/solutions/blue-chip-publications

The dataset covers:
- **Period**: August 1976 - October 2025 (589 monthly observations)
- **Indicators**: Real GDP, Nominal GDP, GDP Deflator, CPI, Industrial Production, Unemployment Rate, Housing Starts, Auto Sales, Corporate Profits

## Models Evaluated

| Family | Models | Win Rate |
|--------|--------|----------|
| **Machine Learning** | XGBoost, RandomForest, LightGBM | 44.4% |
| **Foundation Models** | Chronos-T5-Small | 33.3% |
| **Deep Learning** | LSTM, GRU, Transformer | 22.2% |
| **Extended LSTM** | sLSTM, mLSTM | 0.0% |

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
pip install torch chronos-forecasting
pip install matplotlib seaborn scipy statsmodels
```

## Reproducing Results

1. **Obtain Blue Chip data** from Wolters Kluwer
2. **Run model comparison**:
   ```bash
   python code/17_comprehensive_comparison_all_models.py
   python code/18_final_comprehensive_comparison.py
   ```
3. **Statistical significance testing**:
   ```bash
   python code/19_statistical_significance_testing.py
   ```
4. **Generate publication figures**:
   ```bash
   python code/31_create_publication_figures_v2.py
   ```

## Key Results

### Champion Models by Indicator

| Indicator | Champion Model | MAPE (%) | 95% CI |
|-----------|---------------|----------|--------|
| Industrial Production | XGBoost | 1.87 | [1.52, 2.31] |
| Unemployment Rate | Chronos | 2.55 | [2.11, 3.08] |
| GDP Deflator | XGBoost | 3.01 | [2.45, 3.67] |
| Real GDP | GRU | 3.15 | [2.58, 3.84] |
| CPI | Chronos | 6.21 | [5.12, 7.45] |
| Nominal GDP | XGBoost | 6.75 | [5.52, 8.11] |
| 3-Month Treasury | Transformer | 14.95 | [12.31, 17.89] |
| Housing Starts | XGBoost | 14.95 | [12.22, 18.02] |
| Personal Income | Chronos | 5.11 | [4.23, 6.12] |

### Statistical Significance

- **52.8%** of pairwise comparisons statistically significant (p < 0.05)
- Diebold-Mariano tests with Harvey-Leibourne-Newbold small-sample correction
- Bootstrap confidence intervals (BCa method, 2000 samples)

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{author2025bluechip,
  title={Machine Learning vs Foundation Models for Economic Forecasting:
         Evidence from Blue Chip Indicators},
  author={[Author Names]},
  journal={International Journal of Forecasting},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research, please contact: ktomak@mindspaceai.nl

---

*This repository accompanies a manuscript currently under peer review.*
