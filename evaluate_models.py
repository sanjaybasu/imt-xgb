#!/usr/bin/env python3
"""
IMT-XGB Model Evaluation
========================

This script evaluates the IMT-XGB model against baseline models using the preprocessed data.
It compares performance across all three prediction tasks and generates evaluation metrics.

Author: Healthcare ML Research Team
Date: April 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from datetime import datetime

# Import the IMT-XGB model class
from imt_xgb_model import IMT_XGBoost

def load_preprocessed_data(data_path='data/preprocessed_data_fixed.pkl'):
    """
    Load preprocessed data for model evaluation.
    
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

def evaluate_baseline_models(data, output_dir='results'):
    """
    Evaluate baseline models for comparison with IMT-XGB.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    output_dir : str
        Directory to save evaluation results
        
    Returns:
    --------
    dict
        Dictionary containing baseline model evaluation results
    """
    print("Evaluating baseline models...")
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_event_train = data['y_event_train']
    y_event_test = data['y_event_test']
    y_time_train = data['y_time_train']
    y_time_test = data['y_time_test']
    y_action_train_encoded = data['y_action_train_encoded']
    y_action_test_encoded = data['y_action_test_encoded']
    
    # Initialize results dictionary
    results = {
        'event': {},
        'time': {},
        'action': {}
    }
    
    # 1. Baseline models for event prediction
    print("Training baseline models for event prediction...")
    
    # Logistic Regression
    start_time = time.time()
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_event_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_preds = lr.predict(X_test)
    lr_time = time.time() - start_time
    
    results['event']['logistic_regression'] = {
        'auc': roc_auc_score(y_event_test, lr_probs),
        'precision': precision_score(y_event_test, lr_preds),
        'recall': recall_score(y_event_test, lr_preds),
        'f1': f1_score(y_event_test, lr_preds),
        'training_time': lr_time
    }
    
    # Random Forest
    start_time = time.time()
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_event_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    rf_time = time.time() - start_time
    
    results['event']['random_forest'] = {
        'auc': roc_auc_score(y_event_test, rf_probs),
        'precision': precision_score(y_event_test, rf_preds),
        'recall': recall_score(y_event_test, rf_preds),
        'f1': f1_score(y_event_test, rf_preds),
        'training_time': rf_time
    }
    
    # Standard XGBoost
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_event_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    xgb_time = time.time() - start_time
    
    results['event']['xgboost'] = {
        'auc': roc_auc_score(y_event_test, xgb_probs),
        'precision': precision_score(y_event_test, xgb_preds),
        'recall': recall_score(y_event_test, xgb_preds),
        'f1': f1_score(y_event_test, xgb_preds),
        'training_time': xgb_time
    }
    
    # Save ROC curves for event prediction
    plt.figure(figsize=(10, 8))
    
    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_event_test, lr_probs)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {results["event"]["logistic_regression"]["auc"]:.3f})')
    
    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_event_test, rf_probs)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {results["event"]["random_forest"]["auc"]:.3f})')
    
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_event_test, xgb_probs)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {results["event"]["xgboost"]["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Event Prediction (Baseline Models)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'figures', 'baseline_event_roc_curves.png'), dpi=300, bbox_inches='tight')
    
    # 2. Baseline models for time prediction
    print("Training baseline models for time prediction...")
    
    # Weibull AFT model from lifelines
    start_time = time.time()
    aft = WeibullAFTFitter()
    
    # Prepare data for lifelines
    df_surv = X_train.copy()
    df_surv['time'] = y_time_train
    df_surv['event'] = y_event_train
    
    # Fit model
    aft.fit(df_surv, duration_col='time', event_col='event')
    
    # Predict
    aft_preds = aft.predict_median(X_test)
    aft_time = time.time() - start_time
    
    # Calculate concordance index
    results['time']['weibull_aft'] = {
        'c_index': concordance_index(y_time_test, aft_preds),
        'training_time': aft_time
    }
    
    # 3. Baseline models for action prediction
    print("Training baseline models for CHW action recommendation...")
    
    # Random Forest
    start_time = time.time()
    rf_action = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_action.fit(X_train, y_action_train_encoded)
    rf_action_preds = rf_action.predict(X_test)
    rf_action_time = time.time() - start_time
    
    results['action']['random_forest'] = {
        'accuracy': accuracy_score(y_action_test_encoded, rf_action_preds),
        'f1': f1_score(y_action_test_encoded, rf_action_preds, average='weighted'),
        'training_time': rf_action_time
    }
    
    # Standard XGBoost
    start_time = time.time()
    xgb_action = xgb.XGBClassifier(
        random_state=42, 
        objective='multi:softprob', 
        num_class=len(np.unique(y_action_train_encoded))
    )
    xgb_action.fit(X_train, y_action_train_encoded)
    xgb_action_preds = xgb_action.predict(X_test)
    xgb_action_time = time.time() - start_time
    
    results['action']['xgboost'] = {
        'accuracy': accuracy_score(y_action_test_encoded, xgb_action_preds),
        'f1': f1_score(y_action_test_encoded, xgb_action_preds, average='weighted'),
        'training_time': xgb_action_time
    }
    
    # Save confusion matrix for action prediction (XGBoost)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_action_test_encoded, xgb_action_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - XGBoost (CHW Action Recommendation)')
    
    action_classes = data['action_encoder'].classes_
    plt.xticks(np.arange(len(action_classes)) + 0.5, action_classes, rotation=45)
    plt.yticks(np.arange(len(action_classes)) + 0.5, action_classes, rotation=45)
    
    plt.savefig(os.path.join(output_dir, 'figures', 'baseline_action_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'baseline_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Baseline evaluation results saved to {os.path.join(output_dir, 'baseline_results.pkl')}")
    
    return results

def train_and_evaluate_imt_xgb(data, output_dir='results'):
    """
    Train and evaluate the IMT-XGB model.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing preprocessed data splits
    output_dir : str
        Directory to save evaluation results
        
    Returns:
    --------
    tuple
        (IMT-XGB model, evaluation results)
    """
    print("Training and evaluating IMT-XGB model...")
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_event_train = data['y_event_train']
    y_event_test = data['y_event_test']
    y_time_train = data['y_time_train']
    y_time_test = data['y_time_test']
    y_action_train = data['y_action_train']
    y_action_test = data['y_action_test']
    
    # Define missingness features
    missingness_features = ['primary_care_visits', 'last_pcp_visit_days', 
                           'no_primary_care', 'pcp_visit_recent', 
                           'pcp_visit_moderate', 'pcp_visit_distant']
    
    # Initialize and train IMT-XGB model
    start_time = time.time()
    imt_xgb = IMT_XGBoost(missingness_features=missingness_features, random_state=42)
    imt_xgb.fit(X_train, y_event_train, y_time_train, y_action_train, validation_split=0.2)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    results = imt_xgb.evaluate(X_test, y_event_test, y_time_test, y_action_test)
    
    # Add training time to results
    results['training_time'] = training_time
    
    # Save model
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    imt_xgb.save_model(os.path.join(output_dir, 'models', 'imt_xgb'))
    
    # Save results
    with open(os.path.join(output_dir, 'imt_xgb_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"IMT-XGB evaluation results saved to {os.path.join(output_dir, 'imt_xgb_results.pkl')}")
    
    # Generate ROC curve for event prediction
    dtest = xgb.DMatrix(X_test)
    event_probs = imt_xgb.models['event'].predict(dtest)
    
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_event_test, event_probs)
    plt.plot(fpr, tpr, label=f'IMT-XGB (AUC = {results["event"]["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Event Prediction (IMT-XGB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'figures', 'imt_xgb_event_roc_curve.png'), dpi=300, bbox_inches='tight')
    
    # Generate confusion matrix for action prediction
    predictions = imt_xgb.predict(X_test)
    action_preds = data['action_encoder'].transform(predictions['action_prediction'])
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(data['y_action_test_encoded'], action_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - IMT-XGB (CHW Action Recommendation)')
    
    action_classes = data['action_encoder'].classes_
    plt.xticks(np.arange(len(action_classes)) + 0.5, action_classes, rotation=45)
    plt.yticks(np.arange(len(action_classes)) + 0.5, action_classes, rotation=45)
    
    plt.savefig(os.path.join(output_dir, 'figures', 'imt_xgb_action_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    return imt_xgb, results

def compare_models(baseline_results, imt_xgb_results, output_dir='results'):
    """
    Compare IMT-XGB with baseline models and generate comparison visualizations.
    
    Parameters:
    -----------
    baseline_results : dict
        Dictionary containing baseline model evaluation results
    imt_xgb_results : dict
        Dictionary containing IMT-XGB evaluation results
    output_dir : str
        Directory to save comparison results
    """
    print("Comparing IMT-XGB with baseline models...")
    
    # Combine results
    comparison = {
        'event': {},
        'time': {},
        'action': {}
    }
    
    # Event prediction comparison
    for model in baseline_results['event']:
        comparison['event'][model] = baseline_results['event'][model]
    comparison['event']['imt_xgb'] = imt_xgb_results['event']
    
    # Time prediction comparison
    for model in baseline_results['time']:
        comparison['time'][model] = baseline_results['time'][model]
    comparison['time']['imt_xgb'] = imt_xgb_results['time']
    
    # Action prediction comparison
    for model in baseline_results['action']:
        comparison['action'][model] = baseline_results['action'][model]
    comparison['action']['imt_xgb'] = {
        'accuracy': imt_xgb_results['action']['accuracy'],
        'f1': imt_xgb_results['action']['f1']
    }
    
    # Save comparison results
    with open(os.path.join(output_dir, 'model_comparison.pkl'), 'wb') as f:
        pickle.dump(comparison, f)
    
    print(f"Model comparison results saved to {os.path.join(output_dir, 'model_comparison.pkl')}")
    
    # Generate comparison visualizations
    
    # 1. Event prediction comparison (AUC)
    plt.figure(figsize=(12, 8))
    models = list(comparison['event'].keys())
    auc_values = [comparison['event'][model]['auc'] for model in models]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(models, auc_values, color=colors)
    
    plt.xlabel('Model')
    plt.ylabel('AUC')
    plt.title('Event Prediction Performance Comparison (AUC)')
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'figures', 'event_prediction_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Time prediction comparison (C-index)
    plt.figure(figsize=(12, 8))
    models = list(comparison['time'].keys())
    c_index_values = [comparison['time'][model]['c_index'] for model in models]
    
    bars = plt.bar(models, c_index_values, color=colors[:len(models)])
    
    plt.xlabel('Model')
    plt.ylabel('Concordance Index')
    plt.title('Time-to-Event Prediction Performance Comparison (C-index)')
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'figures', 'time_prediction_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 3. Action prediction comparison (F1 score)
    plt.figure(figsize=(12, 8))
    models = list(comparison['action'].keys())
    f1_values = [comparison['action'][model]['f1'] for model in models]
    
    bars = plt.bar(models, f1_values, color=colors[:len(models)])
    
    plt.xlabel('Model')
    plt.ylabel('F1 Score (weighted)')
    plt.title('CHW Action Recommendation Performance Comparison (F1 Score)')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'figures', 'action_prediction_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 4. Training time comparison
    plt.figure(figsize=(12, 8))
    
    # Get training times
    training_times = {}
    for task in ['event', 'action']:
        for model in comparison[task]:
            if model not in training_times and model != 'imt_xgb':
                if 'training_time' in comparison[task][model]:
                    training_times[model] = comparison[task][model]['training_time']
    
    # Add IMT-XGB training time
    training_times['imt_xgb'] = imt_xgb_results['training_time']
    
    models = list(training_times.keys())
    times = [training_times[model] for model in models]
    
    bars = plt.bar(models, times, color=colors[:len(models)])
    
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'figures', 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Generate summary table
    summary_table = pd.DataFrame({
        'Model': models + ['weibull_aft'],
        'Event AUC': [comparison['event'].get(model, {}).get('auc', float('nan')) for model in models] + [float('nan')],
        'Event F1': [comparison['event'].get(model, {}).get('f1', float('nan')) for model in models] + [float('nan')],
        'Time C-index': [comparison['time'].get(model, {}).get('c_index', float('nan')) for model in models[:-1]] + 
                        [comparison['time']['weibull_aft']['c_index']] + [float('nan')],
        'Action F1': [comparison['action'].get(model, {}).get('f1', float('nan')) for model in models] + [float('nan')],
        'Training Time (s)': [training_times.get(model, float('nan')) for model in models] + 
                            [comparison['time']['weibull_aft'].get('training_time', float('nan'))]
    })
    
    # Save summary table
    summary_table.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
    
    print(f"Model comparison summary saved to {os.path.join(output_dir, 'model_comparison_summary.csv')}")
    
    return comparison

if __name__ == "__main__":
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Load preprocessed data
    data = load_preprocessed_data()
    
    # Evaluate baseline models
    baseline_results = evaluate_baseline_models(data)
    
    # Train and evaluate IMT-XGB model
    imt_xgb, imt_xgb_results = train_and_evaluate_imt_xgb(data)
    
    # Compare models
    comparison = compare_models(baseline_results, imt_xgb_results)
    
    print("Model evaluation complete!")
