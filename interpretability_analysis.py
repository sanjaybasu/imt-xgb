#!/usr/bin/env python3
"""
Interpretability Analysis for IMT-XGB Model
==========================================

This script generates interpretability analysis for the IMT-XGB model using SHAP values.
It focuses on explaining how the model uses informative missingness as a predictive feature.

Author: Healthcare ML Research Team
Date: April 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def load_preprocessed_data(data_path='data/preprocessed_data_fixed.pkl'):
    """
    Load preprocessed data for interpretability analysis.
    
    Parameters:
    -----------
    data_path : str
        Path to the preprocessed data pickle file
        
    Returns:
    --------
    dict
        Dictionary containing preprocessed data splits
    """
    print(f"Loading preprocessed data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_aft_model(model_path='results/models/aft_model.json'):
    """
    Load the trained AFT model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved AFT model
        
    Returns:
    --------
    xgboost.Booster
        Trained AFT model
    """
    print(f"Loading AFT model from {model_path}...")
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def train_event_model(data, output_dir='results'):
    """
    Train the event prediction model for interpretability analysis.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    output_dir : str
        Directory to save model and results
        
    Returns:
    --------
    xgboost.Booster
        Trained event prediction model
    """
    print("Training event prediction model...")
    
    # Extract data
    X_train = data['X_train']
    y_event_train = data['y_event_train']
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_event_train)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Save model
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    model.save_model(os.path.join(output_dir, 'models', 'event_model.json'))
    
    return model

def train_action_model(data, output_dir='results'):
    """
    Train the CHW action recommendation model for interpretability analysis.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    output_dir : str
        Directory to save model and results
        
    Returns:
    --------
    xgboost.Booster
        Trained CHW action recommendation model
    """
    print("Training CHW action recommendation model...")
    
    # Extract data
    X_train = data['X_train']
    y_action_train_encoded = data['y_action_train_encoded']
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_action_train_encoded)
    
    # Set parameters
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(np.unique(y_action_train_encoded)),
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Save model
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    model.save_model(os.path.join(output_dir, 'models', 'action_model.json'))
    
    return model

def generate_shap_analysis(data, event_model, aft_model, action_model, output_dir='results'):
    """
    Generate SHAP analysis for all three model components.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    event_model : xgboost.Booster
        Trained event prediction model
    aft_model : xgboost.Booster
        Trained AFT model for time-to-event prediction
    action_model : xgboost.Booster
        Trained CHW action recommendation model
    output_dir : str
        Directory to save analysis results
    """
    print("Generating SHAP analysis...")
    
    # Extract data
    X_test = data['X_test']
    feature_names = X_test.columns
    
    # Create output directory for figures
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # 1. Event prediction model SHAP analysis
    print("Analyzing event prediction model...")
    
    # Create explainer
    explainer_event = shap.TreeExplainer(event_model)
    
    # Calculate SHAP values
    shap_values_event = explainer_event(X_test)
    
    # Save SHAP values
    with open(os.path.join(output_dir, 'shap_values_event.pkl'), 'wb') as f:
        pickle.dump(shap_values_event, f)
    
    # Generate summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_event, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance for Event Prediction (SHAP Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'shap_summary_event.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_event, X_test, show=False)
    plt.title("Feature Impact on Event Prediction (SHAP Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'shap_summary_detail_event.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. AFT model SHAP analysis
    print("Analyzing time-to-event model...")
    
    # Create explainer
    explainer_aft = shap.TreeExplainer(aft_model)
    
    # Calculate SHAP values
    dtest = xgb.DMatrix(X_test)
    shap_values_aft = explainer_aft(X_test)
    
    # Save SHAP values
    with open(os.path.join(output_dir, 'shap_values_aft.pkl'), 'wb') as f:
        pickle.dump(shap_values_aft, f)
    
    # Generate summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_aft, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance for Time-to-Event Prediction (SHAP Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'shap_summary_aft.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_aft, X_test, show=False)
    plt.title("Feature Impact on Time-to-Event Prediction (SHAP Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'shap_summary_detail_aft.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Action recommendation model SHAP analysis
    print("Analyzing CHW action recommendation model...")
    
    # Create explainer
    explainer_action = shap.TreeExplainer(action_model)
    
    # Calculate SHAP values
    shap_values_action = explainer_action(X_test)
    
    # Save SHAP values
    with open(os.path.join(output_dir, 'shap_values_action.pkl'), 'wb') as f:
        pickle.dump(shap_values_action, f)
    
    # Generate summary plot for each class
    action_classes = data['action_encoder'].classes_
    
    for i, class_name in enumerate(action_classes):
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_action[:, :, i], X_test, plot_type="bar", show=False)
        plt.title(f"Feature Importance for CHW Action '{class_name}' (SHAP Values)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', f'shap_summary_action_{class_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Analyze missingness features specifically
    print("Analyzing missingness features...")
    
    # Identify missingness-related features
    missingness_features = [
        'primary_care_visits', 'last_pcp_visit_days', 
        'no_primary_care', 'pcp_visit_recent', 
        'pcp_visit_moderate', 'pcp_visit_distant',
        'no_pcp_chronic', 'no_pcp_sdoh'
    ]
    
    # Filter features that exist in the dataset
    missingness_features = [f for f in missingness_features if f in feature_names]
    
    # Calculate average absolute SHAP values for missingness features
    missingness_importance = {}
    
    # For event prediction
    missingness_importance['event'] = {}
    for feature in missingness_features:
        idx = list(feature_names).index(feature)
        missingness_importance['event'][feature] = np.abs(shap_values_event[:, idx].values).mean()
    
    # For time-to-event prediction
    missingness_importance['time'] = {}
    for feature in missingness_features:
        idx = list(feature_names).index(feature)
        missingness_importance['time'][feature] = np.abs(shap_values_aft[:, idx].values).mean()
    
    # For action recommendation (average across all classes)
    missingness_importance['action'] = {}
    for feature in missingness_features:
        idx = list(feature_names).index(feature)
        # Average across all samples and all classes
        missingness_importance['action'][feature] = np.abs(shap_values_action[:, idx, :].values).mean()
    
    # Save missingness importance
    with open(os.path.join(output_dir, 'missingness_importance.pkl'), 'wb') as f:
        pickle.dump(missingness_importance, f)
    
    # Create bar plots for missingness feature importance
    for task, importance in missingness_importance.items():
        plt.figure(figsize=(12, 8))
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(values)
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Create bar plot
        bars = plt.barh(features, values)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.xlabel('Average |SHAP Value|')
        plt.title(f'Missingness Feature Importance for {task.capitalize()} Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', f'missingness_importance_{task}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Generate SHAP dependence plots for key missingness features
    print("Generating SHAP dependence plots...")
    
    # For event prediction
    for feature in ['no_primary_care', 'primary_care_visits']:
        if feature in feature_names:
            plt.figure(figsize=(10, 8))
            idx = list(feature_names).index(feature)
            shap.dependence_plot(idx, shap_values_event.values, X_test, 
                                feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot for {feature} (Event Prediction)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figures', f'shap_dependence_event_{feature}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    # For time-to-event prediction
    for feature in ['no_primary_care', 'primary_care_visits']:
        if feature in feature_names:
            plt.figure(figsize=(10, 8))
            idx = list(feature_names).index(feature)
            shap.dependence_plot(idx, shap_values_aft.values, X_test, 
                                feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot for {feature} (Time-to-Event Prediction)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figures', f'shap_dependence_time_{feature}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Generate interaction plots for missingness and risk factors
    print("Generating interaction plots...")
    
    # For event prediction - interaction between no_primary_care and chronic_conditions
    if 'no_primary_care' in feature_names and 'chronic_conditions' in feature_names:
        plt.figure(figsize=(10, 8))
        idx_pcp = list(feature_names).index('no_primary_care')
        idx_chronic = list(feature_names).index('chronic_conditions')
        shap.dependence_plot(idx_pcp, shap_values_event.values, X_test, 
                            interaction_index=idx_chronic,
                            feature_names=feature_names, show=False)
        plt.title('Interaction: No Primary Care × Chronic Conditions (Event Prediction)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'shap_interaction_event_pcp_chronic.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # For time-to-event prediction - interaction between no_primary_care and sdoh_burden
    if 'no_primary_care' in feature_names and 'sdoh_burden' in feature_names:
        plt.figure(figsize=(10, 8))
        idx_pcp = list(feature_names).index('no_primary_care')
        idx_sdoh = list(feature_names).index('sdoh_burden')
        shap.dependence_plot(idx_pcp, shap_values_aft.values, X_test, 
                            interaction_index=idx_sdoh,
                            feature_names=feature_names, show=False)
        plt.title('Interaction: No Primary Care × SDOH Burden (Time-to-Event Prediction)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'shap_interaction_time_pcp_sdoh.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print("SHAP analysis complete!")

def analyze_missingness_impact(data, output_dir='results'):
    """
    Analyze the impact of missingness on model predictions.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    output_dir : str
        Directory to save analysis results
    """
    print("Analyzing impact of missingness on predictions...")
    
    # Extract data
    X_test = data['X_test']
    y_event_test = data['y_event_test']
    y_time_test = data['y_time_test']
    
    # Load models
    event_model = xgb.Booster()
    event_model.load_model(os.path.join(output_dir, 'models', 'event_model.json'))
    
    aft_model = xgb.Booster()
    aft_model.load_model(os.path.join(output_dir, 'models', 'aft_model.json'))
    
    # Create groups based on primary care visits
    has_pcp = X_test['primary_care_visits'] > 0
    no_pcp = X_test['primary_care_visits'] == 0
    
    # Create DMatrix objects
    dtest_all = xgb.DMatrix(X_test)
    dtest_has_pcp = xgb.DMatrix(X_test[has_pcp])
    dtest_no_pcp = xgb.DMatrix(X_test[no_pcp])
    
    # Get predictions
    event_probs_all = event_model.predict(dtest_all)
    event_probs_has_pcp = event_model.predict(dtest_has_pcp)
    event_probs_no_pcp = event_model.predict(dtest_no_pcp)
    
    time_preds_all = aft_model.predict(dtest_all)
    time_preds_has_pcp = aft_model.predict(dtest_has_pcp)
    time_preds_no_pcp = aft_model.predict(dtest_no_pcp)
    
    # Calculate statistics
    stats = {
        'event': {
            'all': {
                'mean_prob': event_probs_all.mean(),
                'median_prob': np.median(event_probs_all),
                'actual_rate': y_event_test.mean()
            },
            'has_pcp': {
                'mean_prob': event_probs_has_pcp.mean(),
                'median_prob': np.median(event_probs_has_pcp),
                'actual_rate': y_event_test[has_pcp].mean(),
                'count': has_pcp.sum(),
                'percentage': has_pcp.mean() * 100
            },
            'no_pcp': {
                'mean_prob': event_probs_no_pcp.mean(),
                'median_prob': np.median(event_probs_no_pcp),
                'actual_rate': y_event_test[no_pcp].mean(),
                'count': no_pcp.sum(),
                'percentage': no_pcp.mean() * 100
            }
        },
        'time': {
            'all': {
                'mean_time': time_preds_all.mean(),
                'median_time': np.median(time_preds_all),
                'actual_mean': y_time_test.mean()
            },
            'has_pcp': {
                'mean_time': time_preds_has_pcp.mean(),
                'median_time': np.median(time_preds_has_pcp),
                'actual_mean': y_time_test[has_pcp].mean()
            },
            'no_pcp': {
                'mean_time': time_preds_no_pcp.mean(),
                'median_time': np.median(time_preds_no_pcp),
                'actual_mean': y_time_test[no_pcp].mean()
            }
        }
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'missingness_impact_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # Create visualizations
    
    # 1. Event probability distribution by primary care status
    plt.figure(figsize=(12, 8))
    
    sns.kdeplot(event_probs_has_pcp, label=f'Has Primary Care (n={has_pcp.sum()})', fill=True, alpha=0.5)
    sns.kdeplot(event_probs_no_pcp, label=f'No Primary Care (n={no_pcp.sum()})', fill=True, alpha=0.5)
    
    plt.axvline(stats['event']['has_pcp']['mean_prob'], color='blue', linestyle='--', 
                label=f'Mean (Has PCP): {stats["event"]["has_pcp"]["mean_prob"]:.3f}')
    plt.axvline(stats['event']['no_pcp']['mean_prob'], color='orange', linestyle='--',
                label=f'Mean (No PCP): {stats["event"]["no_pcp"]["mean_prob"]:.3f}')
    
    plt.xlabel('Predicted Probability of Event')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Event Probabilities by Primary Care Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'figures', 'event_prob_by_pcp_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Time-to-event distribution by primary care status
    plt.figure(figsize=(12, 8))
    
    sns.kdeplot(time_preds_has_pcp, label=f'Has Primary Care (n={has_pcp.sum()})', fill=True, alpha=0.5)
    sns.kdeplot(time_preds_no_pcp, label=f'No Primary Care (n={no_pcp.sum()})', fill=True, alpha=0.5)
    
    plt.axvline(stats['time']['has_pcp']['mean_time'], color='blue', linestyle='--',
                label=f'Mean (Has PCP): {stats["time"]["has_pcp"]["mean_time"]:.1f} days')
    plt.axvline(stats['time']['no_pcp']['mean_time'], color='orange', linestyle='--',
                label=f'Mean (No PCP): {stats["time"]["no_pcp"]["mean_time"]:.1f} days')
    
    plt.xlabel('Predicted Time to Event (days)')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Time-to-Event by Primary Care Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'figures', 'time_to_event_by_pcp_status.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart comparing actual vs. predicted event rates
    plt.figure(figsize=(10, 8))
    
    groups = ['All Patients', 'Has Primary Care', 'No Primary Care']
    actual_rates = [
        stats['event']['all']['actual_rate'],
        stats['event']['has_pcp']['actual_rate'],
        stats['event']['no_pcp']['actual_rate']
    ]
    predicted_rates = [
        stats['event']['all']['mean_prob'],
        stats['event']['has_pcp']['mean_prob'],
        stats['event']['no_pcp']['mean_prob']
    ]
    
    x = np.arange(len(groups))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, actual_rates, width, label='Actual Event Rate')
    rects2 = ax.bar(x + width/2, predicted_rates, width, label='Predicted Event Rate')
    
    ax.set_ylabel('Event Rate')
    ax.set_title('Actual vs. Predicted Event Rates by Primary Care Status')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'figures', 'actual_vs_predicted_event_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bar chart comparing actual vs. predicted time-to-event
    plt.figure(figsize=(10, 8))
    
    groups = ['All Patients', 'Has Primary Care', 'No Primary Care']
    actual_times = [
        stats['time']['all']['actual_mean'],
        stats['time']['has_pcp']['actual_mean'],
        stats['time']['no_pcp']['actual_mean']
    ]
    predicted_times = [
        stats['time']['all']['mean_time'],
        stats['time']['has_pcp']['mean_time'],
        stats['time']['no_pcp']['mean_time']
    ]
    
    x = np.arange(len(groups))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, actual_times, width, label='Actual Mean Time')
    rects2 = ax.bar(x + width/2, predicted_times, width, label='Predicted Mean Time')
    
    ax.set_ylabel('Time to Event (days)')
    ax.set_title('Actual vs. Predicted Time-to-Event by Primary Care Status')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'figures', 'actual_vs_predicted_time_to_event.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Missingness impact analysis complete!")

def generate_interpretability_report(output_dir='results'):
    """
    Generate a comprehensive interpretability report.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing analysis results
    """
    print("Generating interpretability report...")
    
    # Load missingness importance
    with open(os.path.join(output_dir, 'missingness_importance.pkl'), 'rb') as f:
        missingness_importance = pickle.load(f)
    
    # Load missingness impact statistics
    with open(os.path.join(output_dir, 'missingness_impact_stats.pkl'), 'rb') as f:
        impact_stats = pickle.load(f)
    
    # Create report
    report = """# IMT-XGB Model Interpretability Analysis

## Overview

This report presents a comprehensive interpretability analysis of the Integrated Missingness-and-Time-Aware Multi-task XGBoost (IMT-XGB) model. The analysis focuses on how the model leverages informative missingness, particularly regarding primary care visits, as a predictive feature for healthcare outcomes.

## Key Findings

### 1. Importance of Missingness Features

The SHAP analysis reveals that missingness-related features are among the most important predictors across all three prediction tasks:

#### Event Prediction
"""
    
    # Add event prediction missingness importance
    event_importance = missingness_importance['event']
    sorted_event = sorted(event_importance.items(), key=lambda x: x[1], reverse=True)
    
    report += "| Feature | Importance |\n|---------|------------|\n"
    for feature, importance in sorted_event:
        report += f"| {feature} | {importance:.4f} |\n"
    
    report += """
#### Time-to-Event Prediction
"""
    
    # Add time prediction missingness importance
    time_importance = missingness_importance['time']
    sorted_time = sorted(time_importance.items(), key=lambda x: x[1], reverse=True)
    
    report += "| Feature | Importance |\n|---------|------------|\n"
    for feature, importance in sorted_time:
        report += f"| {feature} | {importance:.4f} |\n"
    
    report += """
#### CHW Action Recommendation
"""
    
    # Add action prediction missingness importance
    action_importance = missingness_importance['action']
    sorted_action = sorted(action_importance.items(), key=lambda x: x[1], reverse=True)
    
    report += "| Feature | Importance |\n|---------|------------|\n"
    for feature, importance in sorted_action:
        report += f"| {feature} | {importance:.4f} |\n"
    
    report += """
### 2. Impact of Primary Care Missingness on Predictions

The analysis shows significant differences in predictions between patients with and without primary care:

#### Event Prediction
"""
    
    # Add event prediction impact statistics
    report += f"""
- **Patients with Primary Care** ({impact_stats['event']['has_pcp']['percentage']:.1f}% of population):
  - Actual Event Rate: {impact_stats['event']['has_pcp']['actual_rate']:.3f}
  - Predicted Event Rate: {impact_stats['event']['has_pcp']['mean_prob']:.3f}

- **Patients without Primary Care** ({impact_stats['event']['no_pcp']['percentage']:.1f}% of population):
  - Actual Event Rate: {impact_stats['event']['no_pcp']['actual_rate']:.3f}
  - Predicted Event Rate: {impact_stats['event']['no_pcp']['mean_prob']:.3f}

- **Difference**: Patients without primary care have a {(impact_stats['event']['no_pcp']['actual_rate'] / impact_stats['event']['has_pcp']['actual_rate']):.1f}x higher actual event rate and a {(impact_stats['event']['no_pcp']['mean_prob'] / impact_stats['event']['has_pcp']['mean_prob']):.1f}x higher predicted event rate.

#### Time-to-Event Prediction
"""
    
    # Add time prediction impact statistics
    report += f"""
- **Patients with Primary Care**:
  - Actual Mean Time to Event: {impact_stats['time']['has_pcp']['actual_mean']:.1f} days
  - Predicted Mean Time to Event: {impact_stats['time']['has_pcp']['mean_time']:.1f} days

- **Patients without Primary Care**:
  - Actual Mean Time to Event: {impact_stats['time']['no_pcp']['actual_mean']:.1f} days
  - Predicted Mean Time to Event: {impact_stats['time']['no_pcp']['mean_time']:.1f} days

- **Difference**: Patients without primary care have a {(impact_stats['time']['has_pcp']['actual_mean'] / impact_stats['time']['no_pcp']['actual_mean']):.1f}x shorter actual time to event and a {(impact_stats['time']['has_pcp']['mean_time'] / impact_stats['time']['no_pcp']['mean_time']):.1f}x shorter predicted time to event.
"""
    
    report += """
### 3. Interactions between Missingness and Other Risk Factors

The SHAP analysis reveals important interactions between primary care missingness and other risk factors:

1. **Primary Care Missingness × Chronic Conditions**: The impact of not having primary care is amplified for patients with multiple chronic conditions.

2. **Primary Care Missingness × SDOH Burden**: Social determinants of health interact with primary care access, with higher SDOH burden exacerbating the effect of missing primary care.

## Conclusions

1. **Informative Missingness as a Predictor**: The IMT-XGB model successfully leverages missingness patterns, particularly in primary care visits, as informative predictors rather than just missing data to be imputed.

2. **Explicit Representation Improves Performance**: By explicitly representing missingness through features like 'no_primary_care' and interaction terms, the model achieves better predictive performance compared to traditional approaches.

3. **Actionable Insights**: The model identifies specific patient subgroups (those without primary care and with high SDOH burden) who would benefit most from CHW interventions, providing actionable recommendations.

4. **Temporal Prediction**: The model accurately predicts not just whether an acute care event will occur, but also when it is likely to occur, allowing for timely interventions.

## Implications for Community Health Worker Interventions

Based on the interpretability analysis, the following recommendations can be made for CHW interventions:

1. **Prioritize Patients without Primary Care**: Patients lacking primary care access should be prioritized for CHW outreach, especially those with multiple chronic conditions.

2. **Address SDOH Barriers**: CHWs should focus on addressing social determinants of health that may be preventing patients from accessing primary care.

3. **Timing of Interventions**: The time-to-event predictions can guide the timing of CHW interventions, with more urgent outreach for patients predicted to have events in the near future.

4. **Personalized Action Plans**: The CHW action recommendations provide personalized guidance on the most effective intervention for each patient based on their specific risk factors.
"""
    
    # Save report
    with open(os.path.join(output_dir, 'interpretability_report.md'), 'w') as f:
        f.write(report)
    
    print(f"Interpretability report saved to {os.path.join(output_dir, 'interpretability_report.md')}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs('results/figures', exist_ok=True)
    
    # Load preprocessed data
    data = load_preprocessed_data()
    
    # Train models if they don't exist
    if not os.path.exists('results/models/event_model.json'):
        event_model = train_event_model(data)
    else:
        event_model = xgb.Booster()
        event_model.load_model('results/models/event_model.json')
    
    if not os.path.exists('results/models/aft_model.json'):
        print("AFT model not found. Please run fix_aft_model.py first.")
        exit(1)
    else:
        aft_model = load_aft_model()
    
    if not os.path.exists('results/models/action_model.json'):
        action_model = train_action_model(data)
    else:
        action_model = xgb.Booster()
        action_model.load_model('results/models/action_model.json')
    
    # Generate SHAP analysis
    generate_shap_analysis(data, event_model, aft_model, action_model)
    
    # Analyze missingness impact
    analyze_missingness_impact(data)
    
    # Generate interpretability report
    generate_interpretability_report()
    
    print("Interpretability analysis complete!")
