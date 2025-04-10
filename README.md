# IMT-XGB: Reproduce Results

This script reproduces the main results from the paper "Integrated Missingness-and-Time-Aware Multi-task XGBoost (IMT-XGB): A Novel Machine Learning Approach for Claims Data".

## Usage

```bash
python reproduce_results.py
```

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- CMS Synthetic Medicare Claims Data (see Data Access section)

## Data Access

The CMS Synthetic Medicare Claims Data can be obtained from:
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs

## Output

This script will generate:
1. Model performance metrics (AUC, C-index, F1-score)
2. Clinical impact metrics (NNT, NNH)
3. SHAP analysis visualizations
4. Subgroup performance analysis
