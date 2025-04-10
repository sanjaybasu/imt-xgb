"""
Reproduce Results for IMT-XGB Paper

This script reproduces the main results from the paper:
"Integrated Missingness-and-Time-Aware Multi-task XGBoost (IMT-XGB): 
A Novel Machine Learning Approach for Medicare Claims Data"

Author: Sanjay Basu, MD, PhD
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# Import custom modules
from imt_xgb_model import IMT_XGB
from evaluate_models import evaluate_all_models
from clinical_metrics import calculate_nnt_nnh
from confidence_intervals import bootstrap_confidence_intervals
from interpretability_analysis import shap_analysis

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("Reproducing results for IMT-XGB paper...")
    
    # Step 1: Load and prepare data
    print("Step 1: Loading CMS Synthetic Medicare Claims Data...")
    # Note: Users need to download this data from CMS website
    # and update the path accordingly
    data_path = "path/to/cms_synthetic_data.csv"
    
    # If data file doesn't exist, provide instructions
    if not os.path.exists(data_path):
        print("Data file not found. Please download the CMS Synthetic Medicare Claims Data from:")
        print("https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs")
        print("and update the data_path variable in this script.")
        return
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Step 2: Train and evaluate models
    print("Step 2: Training and evaluating models...")
    results = evaluate_all_models(data)
    
    # Step 3: Calculate clinical impact metrics
    print("Step 3: Calculating clinical impact metrics (NNT, NNH)...")
    clinical_impact = calculate_nnt_nnh(results['imt_xgb_model'], data)
    
    # Step 4: Generate SHAP analysis
    print("Step 4: Generating SHAP analysis...")
    shap_results = shap_analysis(results['imt_xgb_model'], data)
    
    # Step 5: Calculate confidence intervals
    print("Step 5: Calculating bootstrap confidence intervals...")
    confidence_intervals = bootstrap_confidence_intervals(results, data, n_bootstrap=1000)
    
    # Step 6: Generate figures
    print("Step 6: Generating figures...")
    generate_figures(results, clinical_impact, shap_results, confidence_intervals)
    
    print("Results successfully reproduced. Figures saved to 'figures/' directory.")

def generate_figures(results, clinical_impact, shap_results, confidence_intervals):
    """Generate all figures from the paper"""
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Model Architecture (created separately)
    
    # Figure 2: Performance Comparison
    plt.figure(figsize=(15, 5))
    
    # Figure 2A: ROC curves for event prediction
    plt.subplot(1, 3, 1)
    for model_name, model_results in results.items():
        if 'roc_curve' in model_results:
            plt.plot(model_results['roc_curve'][0], model_results['roc_curve'][1], 
                     label=f"{model_name} (AUC={model_results['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('A: Event Prediction (ROC Curves)')
    plt.legend()
    
    # Figure 2B: Kaplan-Meier curves for time-to-event
    plt.subplot(1, 3, 2)
    for model_name, model_results in results.items():
        if 'km_curve' in model_results:
            plt.step(model_results['km_curve'][0], model_results['km_curve'][1], 
                    label=f"{model_name} (C-index={model_results['c_index']:.2f})")
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.title('B: Time-to-Event Prediction')
    plt.legend()
    
    # Figure 2C: F1-scores for intervention recommendation
    plt.subplot(1, 3, 3)
    model_names = []
    f1_scores = []
    error_bars = []
    
    for model_name, model_results in results.items():
        if 'f1_score' in model_results:
            model_names.append(model_name)
            f1_scores.append(model_results['f1_score'])
            error_bars.append([
                model_results['f1_score'] - confidence_intervals[model_name]['f1_score'][0],
                confidence_intervals[model_name]['f1_score'][1] - model_results['f1_score']
            ])
    
    plt.bar(model_names, f1_scores, yerr=np.array(error_bars).T, capsize=10)
    plt.xlabel('Model')
    plt.ylabel('F1-Score')
    plt.title('C: Intervention Recommendation')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/figure2_performance_comparison.png', dpi=300)
    plt.close()
    
    # Figure 3: NNT for Care Management Team Interventions
    plt.figure(figsize=(10, 6))
    
    interventions = list(clinical_impact['nnt'].keys())
    nnt_values = [clinical_impact['nnt'][intervention] for intervention in interventions]
    
    # Sort by NNT (ascending)
    sorted_indices = np.argsort(nnt_values)
    sorted_interventions = [interventions[i] for i in sorted_indices]
    sorted_nnt = [nnt_values[i] for i in sorted_indices]
    
    # Error bars from confidence intervals
    error_bars = []
    for intervention in sorted_interventions:
        ci = clinical_impact['nnt_ci'][intervention]
        error_bars.append([sorted_nnt[sorted_interventions.index(intervention)] - ci[0], 
                          ci[1] - sorted_nnt[sorted_interventions.index(intervention)]])
    
    plt.bar(sorted_interventions, sorted_nnt, yerr=np.array(error_bars).T, capsize=10)
    plt.xlabel('Intervention')
    plt.ylabel('Number Needed to Treat (NNT)')
    plt.title('Number Needed to Treat for Care Management Team Interventions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figures/figure3_nnt_visualization.png', dpi=300)
    plt.close()
    
    # Figure 4: SHAP Analysis (multi-panel)
    plt.figure(figsize=(15, 15))
    
    # Figure 4A: Feature importance
    plt.subplot(2, 2, 1)
    feature_names = shap_results['feature_importance']['feature']
    importance_values = shap_results['feature_importance']['importance']
    
    # Sort by importance (descending)
    sorted_indices = np.argsort(importance_values)[::-1][:10]  # Top 10 features
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = [importance_values[i] for i in sorted_indices]
    
    plt.barh(sorted_features, sorted_importance)
    plt.xlabel('Mean |SHAP| Value')
    plt.title('A: Feature Importance')
    
    # Figure 4B: SHAP dependence plot
    plt.subplot(2, 2, 2)
    # Simplified representation of SHAP dependence plot
    x = shap_results['dependence_plot']['x']
    y = shap_results['dependence_plot']['y']
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Primary Care Visits')
    plt.ylabel('SHAP Value')
    plt.title('B: SHAP Dependence Plot for Primary Care')
    
    # Figure 4C: SHAP interaction plot
    plt.subplot(2, 2, 3)
    # Simplified representation of SHAP interaction plot
    interaction_data = shap_results['interaction_plot']
    plt.scatter(interaction_data['x'], interaction_data['y'], 
               c=interaction_data['color'], cmap='viridis', alpha=0.5)
    plt.xlabel('No Primary Care')
    plt.ylabel('SHAP Value')
    plt.title('C: Interaction between No Primary Care and Chronic Conditions')
    plt.colorbar(label='Chronic Condition Count')
    
    plt.tight_layout()
    plt.savefig('figures/figure4_shap_analysis.png', dpi=300)
    plt.close()
    
    # Figure 5: Detailed Performance Analysis (multi-panel)
    plt.figure(figsize=(15, 7))
    
    # Figure 5A: Confusion matrix
    plt.subplot(1, 2, 1)
    cm = results['imt_xgb_model']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=clinical_impact['interventions'],
               yticklabels=clinical_impact['interventions'])
    plt.xlabel('Predicted Intervention')
    plt.ylabel('True Intervention')
    plt.title('A: Confusion Matrix for Intervention Recommendation')
    
    # Figure 5B: Subgroup analysis
    plt.subplot(1, 2, 2)
    subgroups = results['subgroup_analysis']['subgroup']
    auc_values = results['subgroup_analysis']['auc']
    
    # Error bars from confidence intervals
    error_bars = []
    for subgroup in subgroups:
        ci = confidence_intervals['subgroups'][subgroup]
        error_bars.append([auc_values[subgroups.index(subgroup)] - ci[0], 
                          ci[1] - auc_values[subgroups.index(subgroup)]])
    
    plt.bar(subgroups, auc_values, yerr=np.array(error_bars).T, capsize=10)
    plt.xlabel('Subgroup')
    plt.ylabel('AUC')
    plt.title('B: Subgroup Performance Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('figures/figure5_detailed_analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
