#!/usr/bin/env python3
"""
Confidence Interval Utilities for IMT-XGB Model Evaluation
=========================================================

This module provides functions for calculating 95% confidence intervals
for various model performance metrics using bootstrap resampling.

Author: Sanjay Basu MD PhD, Waymark and University of California San Francisco
Date: April 2025
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from lifelines.utils import concordance_index
from scipy import stats

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, ci=0.95, **kwargs):
    """
    Calculate bootstrap confidence intervals for a given metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels or values
    y_pred : array-like
        Predicted labels or values
    metric_func : function
        Function to calculate the metric (e.g., roc_auc_score, precision_score)
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    **kwargs : dict
        Additional arguments to pass to metric_func
        
    Returns:
    --------
    tuple
        (point_estimate, lower_bound, upper_bound)
    """
    # Calculate point estimate
    point_estimate = metric_func(y_true, y_pred, **kwargs)
    
    # Initialize array to store bootstrap results
    bootstrap_results = np.zeros(n_bootstraps)
    
    # Generate bootstrap samples and calculate metric for each
    rng = np.random.RandomState(42)  # For reproducibility
    for i in range(n_bootstraps):
        # Generate bootstrap sample indices
        indices = rng.randint(0, len(y_true), len(y_true))
        
        # Extract bootstrap samples
        y_true_bootstrap = np.array(y_true)[indices]
        y_pred_bootstrap = np.array(y_pred)[indices]
        
        # Calculate metric for bootstrap sample
        try:
            bootstrap_results[i] = metric_func(y_true_bootstrap, y_pred_bootstrap, **kwargs)
        except:
            # Handle edge cases (e.g., only one class in bootstrap sample)
            bootstrap_results[i] = np.nan
    
    # Remove NaN values
    bootstrap_results = bootstrap_results[~np.isnan(bootstrap_results)]
    
    # Calculate confidence interval bounds
    alpha = (1 - ci) / 2
    lower_bound = np.percentile(bootstrap_results, alpha * 100)
    upper_bound = np.percentile(bootstrap_results, (1 - alpha) * 100)
    
    return point_estimate, lower_bound, upper_bound

def bootstrap_ci_auc(y_true, y_pred_proba, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for AUC.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (auc_point_estimate, lower_bound, upper_bound)
    """
    return bootstrap_ci(y_true, y_pred_proba, roc_auc_score, n_bootstraps, ci)

def bootstrap_ci_precision(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for precision.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (precision_point_estimate, lower_bound, upper_bound)
    """
    return bootstrap_ci(y_true, y_pred, precision_score, n_bootstraps, ci)

def bootstrap_ci_recall(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for recall.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (recall_point_estimate, lower_bound, upper_bound)
    """
    return bootstrap_ci(y_true, y_pred, recall_score, n_bootstraps, ci)

def bootstrap_ci_f1(y_true, y_pred, n_bootstraps=1000, ci=0.95, average='binary'):
    """
    Calculate bootstrap confidence intervals for F1 score.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    average : str
        Averaging method for multiclass F1 (default: 'binary')
        
    Returns:
    --------
    tuple
        (f1_point_estimate, lower_bound, upper_bound)
    """
    return bootstrap_ci(y_true, y_pred, f1_score, n_bootstraps, ci, average=average)

def bootstrap_ci_accuracy(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (accuracy_point_estimate, lower_bound, upper_bound)
    """
    return bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstraps, ci)

def bootstrap_ci_c_index(event_times, predicted_times, event_observed, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for concordance index.
    
    Parameters:
    -----------
    event_times : array-like
        Actual event times
    predicted_times : array-like
        Predicted event times
    event_observed : array-like
        Binary indicator of whether the event was observed (1) or censored (0)
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (c_index_point_estimate, lower_bound, upper_bound)
    """
    # Calculate point estimate
    c_index = concordance_index(event_times, predicted_times, event_observed)
    
    # Initialize array to store bootstrap results
    bootstrap_results = np.zeros(n_bootstraps)
    
    # Generate bootstrap samples and calculate c-index for each
    rng = np.random.RandomState(42)  # For reproducibility
    for i in range(n_bootstraps):
        # Generate bootstrap sample indices
        indices = rng.randint(0, len(event_times), len(event_times))
        
        # Extract bootstrap samples
        times_bootstrap = np.array(event_times)[indices]
        pred_bootstrap = np.array(predicted_times)[indices]
        observed_bootstrap = np.array(event_observed)[indices]
        
        # Calculate c-index for bootstrap sample
        try:
            bootstrap_results[i] = concordance_index(times_bootstrap, pred_bootstrap, observed_bootstrap)
        except:
            # Handle edge cases
            bootstrap_results[i] = np.nan
    
    # Remove NaN values
    bootstrap_results = bootstrap_results[~np.isnan(bootstrap_results)]
    
    # Calculate confidence interval bounds
    alpha = (1 - ci) / 2
    lower_bound = np.percentile(bootstrap_results, alpha * 100)
    upper_bound = np.percentile(bootstrap_results, (1 - alpha) * 100)
    
    return c_index, lower_bound, upper_bound

def format_ci(point_estimate, lower_bound, upper_bound, decimal_places=3):
    """
    Format confidence interval as a string.
    
    Parameters:
    -----------
    point_estimate : float
        Point estimate of the metric
    lower_bound : float
        Lower bound of the confidence interval
    upper_bound : float
        Upper bound of the confidence interval
    decimal_places : int
        Number of decimal places to include
        
    Returns:
    --------
    str
        Formatted string with point estimate and confidence interval
    """
    format_str = f"{{:.{decimal_places}f}} (95% CI: {{:.{decimal_places}f}}-{{:.{decimal_places}f}})"
    return format_str.format(point_estimate, lower_bound, upper_bound)

def format_ci_dict(ci_dict, decimal_places=3):
    """
    Format a dictionary of confidence intervals.
    
    Parameters:
    -----------
    ci_dict : dict
        Dictionary with metric names as keys and (point_estimate, lower_bound, upper_bound) tuples as values
    decimal_places : int
        Number of decimal places to include
        
    Returns:
    --------
    dict
        Dictionary with formatted confidence interval strings
    """
    formatted_dict = {}
    for metric, (point, lower, upper) in ci_dict.items():
        formatted_dict[metric] = format_ci(point, lower, upper, decimal_places)
    return formatted_dict
