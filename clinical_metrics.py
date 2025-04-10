#!/usr/bin/env python3
"""
Clinical Metrics Utilities for IMT-XGB Model Evaluation
======================================================

This module provides functions for calculating clinically relevant metrics
such as Number Needed to Treat (NNT) and Number Needed to Harm (NNH)
for CHW action recommendations.

Author: Sanjay Basu MD PhD, Waymark and University of California San Francisco
Date: April 2025
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from confidence_intervals import bootstrap_ci

def calculate_arr(y_true, y_pred, positive_class=1):
    """
    Calculate Absolute Risk Reduction (ARR) for a binary outcome.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels or recommended actions
    positive_class : int or str
        The class label that represents the positive outcome
        
    Returns:
    --------
    float
        Absolute Risk Reduction (can be negative, indicating risk increase)
    """
    # Create binary masks for treatment and control groups
    treatment_mask = (y_pred == positive_class)
    control_mask = (y_pred != positive_class)
    
    # Calculate event rates in each group
    if sum(treatment_mask) > 0:
        treatment_event_rate = sum((y_true == positive_class) & treatment_mask) / sum(treatment_mask)
    else:
        treatment_event_rate = 0
        
    if sum(control_mask) > 0:
        control_event_rate = sum((y_true == positive_class) & control_mask) / sum(control_mask)
    else:
        control_event_rate = 0
    
    # Calculate absolute risk reduction
    arr = control_event_rate - treatment_event_rate
    
    return arr

def calculate_nnt(y_true, y_pred, positive_class=1, ci=0.95, n_bootstraps=1000):
    """
    Calculate Number Needed to Treat (NNT) with confidence intervals.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels or recommended actions
    positive_class : int or str
        The class label that represents the positive outcome
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    n_bootstraps : int
        Number of bootstrap samples for confidence interval calculation
        
    Returns:
    --------
    tuple
        (nnt_point_estimate, lower_bound, upper_bound)
    """
    # Calculate ARR point estimate
    arr = calculate_arr(y_true, y_pred, positive_class)
    
    # Handle edge cases
    if arr == 0:
        return float('inf'), float('inf'), float('inf')
    
    # Calculate NNT point estimate (round up to nearest whole number)
    nnt = np.ceil(1 / abs(arr))
    
    # Add sign based on whether it's truly NNT (positive ARR) or NNH (negative ARR)
    if arr < 0:
        nnt = -nnt  # This is actually NNH
    
    # Define function to calculate NNT for bootstrap samples
    def nnt_func(y_true, y_pred):
        arr = calculate_arr(y_true, y_pred, positive_class)
        if arr == 0:
            return float('inf')
        nnt = np.ceil(1 / abs(arr))
        if arr < 0:
            nnt = -nnt
        return nnt
    
    # Calculate confidence intervals using bootstrap
    try:
        _, lower_bound, upper_bound = bootstrap_ci(y_true, y_pred, nnt_func, n_bootstraps, ci)
        
        # Ensure bounds are properly ordered (for NNH, lower bound is more negative)
        if nnt < 0:  # NNH case
            lower_bound, upper_bound = upper_bound, lower_bound
            
        return nnt, lower_bound, upper_bound
    except:
        # Handle edge cases in bootstrap
        return nnt, nnt, nnt

def calculate_nnh(y_true, y_pred, negative_class=0, ci=0.95, n_bootstraps=1000):
    """
    Calculate Number Needed to Harm (NNH) with confidence intervals.
    This is essentially NNT for the negative outcome.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels or recommended actions
    negative_class : int or str
        The class label that represents the negative outcome
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    n_bootstraps : int
        Number of bootstrap samples for confidence interval calculation
        
    Returns:
    --------
    tuple
        (nnh_point_estimate, lower_bound, upper_bound)
    """
    # For NNH, we're looking at the negative outcome, so we invert the ARR calculation
    # by treating the negative class as the "positive" outcome for ARR calculation
    nnt, lower, upper = calculate_nnt(y_true, y_pred, negative_class, ci, n_bootstraps)
    
    # If NNT is positive, it means the intervention increases negative outcomes (harm)
    # If NNT is negative, it means the intervention decreases negative outcomes (benefit)
    # For NNH, we want positive values to represent harm, so we negate
    nnh = -nnt
    
    # Swap and negate bounds to maintain proper ordering
    return nnh, -upper, -lower

def calculate_clinical_metrics_for_chw_actions(y_true, y_pred, action_classes, ci=0.95, n_bootstraps=1000):
    """
    Calculate clinically relevant metrics (NNT, NNH) for each CHW action recommendation.
    
    Parameters:
    -----------
    y_true : array-like
        True outcomes (1 for event occurred, 0 for no event)
    y_pred : array-like
        Predicted CHW actions
    action_classes : list
        List of possible CHW action classes
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    n_bootstraps : int
        Number of bootstrap samples for confidence interval calculation
        
    Returns:
    --------
    dict
        Dictionary containing NNT and NNH for each CHW action
    """
    results = {}
    
    # For each action class, calculate NNT and NNH
    for action in action_classes:
        # Create binary prediction array (1 if this action, 0 otherwise)
        binary_pred = np.array([1 if p == action else 0 for p in y_pred])
        
        # Calculate NNT (to prevent event)
        nnt, nnt_lower, nnt_upper = calculate_nnt(y_true, binary_pred, positive_class=0, ci=ci, n_bootstraps=n_bootstraps)
        
        # Calculate NNH (to cause event)
        nnh, nnh_lower, nnh_upper = calculate_nnh(y_true, binary_pred, negative_class=1, ci=ci, n_bootstraps=n_bootstraps)
        
        results[action] = {
            'nnt': nnt,
            'nnt_ci': (nnt_lower, nnt_upper),
            'nnt_formatted': format_nnt(nnt, nnt_lower, nnt_upper),
            'nnh': nnh,
            'nnh_ci': (nnh_lower, nnh_upper),
            'nnh_formatted': format_nnt(nnh, nnh_lower, nnh_upper)
        }
    
    return results

def format_nnt(nnt, lower, upper, decimal_places=1):
    """
    Format NNT/NNH with confidence interval as a string.
    
    Parameters:
    -----------
    nnt : float
        Point estimate of NNT or NNH
    lower : float
        Lower bound of the confidence interval
    upper : float
        Upper bound of the confidence interval
    decimal_places : int
        Number of decimal places to include
        
    Returns:
    --------
    str
        Formatted string with NNT/NNH and confidence interval
    """
    # Handle infinity cases
    if np.isinf(nnt):
        return "∞"
    
    # Format based on whether it's NNT or NNH
    if nnt >= 0:
        prefix = "NNT: "
    else:
        prefix = "NNH: "
        nnt = abs(nnt)
        lower, upper = abs(lower), abs(upper)
        # Swap bounds since NNH is negative NNT
        lower, upper = upper, lower
    
    # Handle infinity in bounds
    if np.isinf(lower) and np.isinf(upper):
        ci_str = "∞"
    elif np.isinf(lower):
        ci_str = f">{upper:.{decimal_places}f}"
    elif np.isinf(upper):
        ci_str = f">{lower:.{decimal_places}f}"
    else:
        ci_str = f"{lower:.{decimal_places}f}-{upper:.{decimal_places}f}"
    
    return f"{prefix}{nnt:.{decimal_places}f} (95% CI: {ci_str})"

def calculate_clinical_impact_table(y_true, y_pred, action_classes, ci=0.95, n_bootstraps=1000):
    """
    Create a clinical impact table for CHW actions showing NNT, NNH, and other metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True outcomes (1 for event occurred, 0 for no event)
    y_pred : array-like
        Predicted CHW actions
    action_classes : list
        List of possible CHW action classes
    ci : float
        Confidence interval level (default: 0.95 for 95% CI)
    n_bootstraps : int
        Number of bootstrap samples for confidence interval calculation
        
    Returns:
    --------
    dict
        Dictionary containing clinical impact metrics for each CHW action
    """
    impact_table = {}
    
    # For each action class, calculate clinical impact metrics
    for action in action_classes:
        # Create binary prediction array (1 if this action, 0 otherwise)
        binary_pred = np.array([1 if p == action else 0 for p in y_pred])
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, binary_pred, labels=[0, 1]).ravel()
        
        # Calculate rates
        if (tp + fp) > 0:
            ppv = tp / (tp + fp)  # Positive predictive value
        else:
            ppv = 0
            
        if (tn + fn) > 0:
            npv = tn / (tn + fn)  # Negative predictive value
        else:
            npv = 0
            
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0
            
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        
        # Calculate NNT and NNH
        nnt, nnt_lower, nnt_upper = calculate_nnt(y_true, binary_pred, positive_class=0, ci=ci, n_bootstraps=n_bootstraps)
        nnh, nnh_lower, nnh_upper = calculate_nnh(y_true, binary_pred, negative_class=1, ci=ci, n_bootstraps=n_bootstraps)
        
        # Calculate absolute risk reduction
        arr = calculate_arr(y_true, binary_pred, positive_class=0)
        
        impact_table[action] = {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'ppv': ppv,
            'npv': npv,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'absolute_risk_reduction': arr,
            'nnt': nnt,
            'nnt_ci': (nnt_lower, nnt_upper),
            'nnt_formatted': format_nnt(nnt, nnt_lower, nnt_upper),
            'nnh': nnh,
            'nnh_ci': (nnh_lower, nnh_upper),
            'nnh_formatted': format_nnt(nnh, nnh_lower, nnh_upper)
        }
    
    return impact_table
