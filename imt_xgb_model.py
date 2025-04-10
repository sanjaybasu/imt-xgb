#!/usr/bin/env python3
"""
Integrated Missingness-and-Time-Aware Multi-task XGBoost (IMT-XGB)
==================================================================

This module implements the novel IMT-XGB approach for healthcare claims data prediction.
The model integrates three key components:
1. Explicit handling of informative missingness in primary care visits
2. Temporal prediction capability through survival analysis
3. Actionable CHW recommendation through multi-class classification

Author: Healthcare ML Research Team
Date: April 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from lifelines.utils import concordance_index
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

class IMT_XGBoost:
    """
    Integrated Missingness-and-Time-Aware Multi-task XGBoost (IMT-XGB)
    
    A novel machine learning approach that explicitly integrates informative missingness
    and provides multi-task prediction capabilities for healthcare claims data.
    """
    
    def __init__(self, 
                 missingness_features=None,
                 event_params=None, 
                 time_params=None, 
                 action_params=None,
                 random_state=42):
        """
        Initialize the IMT-XGB model.
        
        Parameters:
        -----------
        missingness_features : list
            List of feature names that may contain informative missingness
        event_params : dict
            Parameters for the event prediction XGBoost model
        time_params : dict
            Parameters for the time-to-event XGBoost model
        action_params : dict
            Parameters for the CHW action recommendation XGBoost model
        random_state : int
            Random seed for reproducibility
        """
        self.missingness_features = missingness_features or ['primary_care_visits', 'last_pcp_visit_days']
        
        # Default parameters for each component
        self.event_params = event_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'seed': random_state
        }
        
        self.time_params = time_params or {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state
        }
        
        self.action_params = action_params or {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': random_state
        }
        
        self.random_state = random_state
        self.models = {
            'event': None,
            'time': None,
            'action': None
        }
        
        self.label_encoder = None
        self.feature_names = None
        self.action_classes = None
        
    def _preprocess_features(self, X, is_training=True):
        """
        Preprocess features with explicit handling of informative missingness.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        is_training : bool
            Whether this is for training or prediction
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed features with explicit missingness indicators
        """
        X_processed = X.copy()
        
        # Store feature names for later use
        if is_training:
            self.feature_names = list(X_processed.columns)
        
        # Handle missing values in non-missingness features
        for col in X_processed.columns:
            if col not in self.missingness_features:
                # For numerical features, impute with median
                if X_processed[col].dtype != 'object':
                    if is_training:
                        # Store median for later use
                        self._medians = self._medians if hasattr(self, '_medians') else {}
                        self._medians[col] = X_processed[col].median()
                    
                    # Impute missing values
                    X_processed[col] = X_processed[col].fillna(self._medians.get(col, 0))
                    
                    # Create missingness indicator
                    X_processed[f'{col}_missing'] = X[col].isna().astype(int)
                
                # For categorical features, impute with most frequent value
                else:
                    if is_training:
                        # Store most frequent value for later use
                        self._most_frequent = self._most_frequent if hasattr(self, '_most_frequent') else {}
                        self._most_frequent[col] = X_processed[col].mode()[0]
                    
                    # Impute missing values
                    X_processed[col] = X_processed[col].fillna(self._most_frequent.get(col, 'Unknown'))
                    
                    # Create missingness indicator
                    X_processed[f'{col}_missing'] = X[col].isna().astype(int)
        
        # Explicit handling of informative missingness in primary care features
        for col in self.missingness_features:
            if col in X_processed.columns:
                # For primary_care_visits, create binary indicator of no visits
                if col == 'primary_care_visits':
                    X_processed['no_primary_care'] = (X_processed[col] == 0).astype(int)
                
                # For last_pcp_visit_days, create indicators for different time ranges
                if col == 'last_pcp_visit_days':
                    X_processed['pcp_visit_recent'] = (X_processed[col] <= 90).astype(int)
                    X_processed['pcp_visit_moderate'] = ((X_processed[col] > 90) & (X_processed[col] <= 180)).astype(int)
                    X_processed['pcp_visit_distant'] = (X_processed[col] > 180).astype(int)
        
        # Create interaction terms between missingness and other risk factors
        X_processed['no_pcp_chronic'] = X_processed.get('no_primary_care', 0) * X_processed.get('chronic_conditions', 0)
        
        # Create SDOH burden score if components are available
        sdoh_cols = ['housing_instability', 'food_insecurity', 'transportation_issues', 'social_isolation']
        if all(col in X_processed.columns for col in sdoh_cols):
            X_processed['sdoh_burden'] = X_processed[sdoh_cols].sum(axis=1)
            X_processed['no_pcp_sdoh'] = X_processed.get('no_primary_care', 0) * X_processed['sdoh_burden']
        
        return X_processed
    
    def fit(self, X, y_event, y_time, y_action, validation_split=0.2):
        """
        Fit the IMT-XGB model to the training data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        y_event : pandas.Series
            Binary target for event occurrence prediction
        y_time : pandas.Series
            Time-to-event target for survival analysis
        y_action : pandas.Series
            Categorical target for CHW action recommendation
        validation_split : float
            Proportion of data to use for validation
            
        Returns:
        --------
        self : object
            Returns self
        """
        print("Fitting IMT-XGB model...")
        
        # Preprocess features
        X_processed = self._preprocess_features(X, is_training=True)
        
        # Encode CHW action labels
        self.label_encoder = LabelEncoder()
        y_action_encoded = self.label_encoder.fit_transform(y_action)
        self.action_classes = self.label_encoder.classes_
        
        # Update action params with number of classes
        self.action_params['num_class'] = len(self.action_classes)
        
        # Split data into training and validation sets
        X_train, X_val, ye_train, ye_val, yt_train, yt_val, ya_train, ya_val = train_test_split(
            X_processed, y_event, y_time, y_action_encoded, 
            test_size=validation_split, 
            random_state=self.random_state,
            stratify=y_event  # Stratify by event occurrence to maintain class balance
        )
        
        print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # 1. Train event occurrence model (binary classification)
        print("Training event occurrence model...")
        dtrain_event = xgb.DMatrix(X_train, label=ye_train)
        dval_event = xgb.DMatrix(X_val, label=ye_val)
        
        watchlist_event = [(dtrain_event, 'train'), (dval_event, 'eval')]
        self.models['event'] = xgb.train(
            self.event_params, 
            dtrain_event, 
            num_boost_round=1000,
            evals=watchlist_event,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 2. Train time-to-event model (survival analysis)
        print("Training time-to-event model...")
        dtrain_time = xgb.DMatrix(X_train, label=yt_train)
        dval_time = xgb.DMatrix(X_val, label=yt_val)
        
        watchlist_time = [(dtrain_time, 'train'), (dval_time, 'eval')]
        self.models['time'] = xgb.train(
            self.time_params, 
            dtrain_time, 
            num_boost_round=1000,
            evals=watchlist_time,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 3. Train CHW action recommendation model (multi-class classification)
        print("Training CHW action recommendation model...")
        dtrain_action = xgb.DMatrix(X_train, label=ya_train)
        dval_action = xgb.DMatrix(X_val, label=ya_val)
        
        watchlist_action = [(dtrain_action, 'train'), (dval_action, 'eval')]
        self.models['action'] = xgb.train(
            self.action_params, 
            dtrain_action, 
            num_boost_round=1000,
            evals=watchlist_action,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Evaluate on validation set
        self._evaluate_validation(X_val, ye_val, yt_val, ya_val)
        
        return self
    
    def _evaluate_validation(self, X_val, y_event_val, y_time_val, y_action_val):
        """
        Evaluate the model on the validation set.
        
        Parameters:
        -----------
        X_val : pandas.DataFrame
            Validation features
        y_event_val : pandas.Series
            Validation binary target
        y_time_val : pandas.Series
            Validation time-to-event target
        y_action_val : pandas.Series
            Validation CHW action target (encoded)
        """
        # Make predictions
        dval = xgb.DMatrix(X_val)
        
        # Event prediction
        event_probs = self.models['event'].predict(dval)
        event_preds = (event_probs > 0.5).astype(int)
        
        # Time prediction
        time_preds = self.models['time'].predict(dval)
        
        # Action prediction
        action_probs = self.models['action'].predict(dval)
        action_preds = np.argmax(action_probs, axis=1)
        
        # Calculate metrics
        # 1. Event prediction metrics
        event_auc = roc_auc_score(y_event_val, event_probs)
        event_precision = precision_score(y_event_val, event_preds)
        event_recall = recall_score(y_event_val, event_preds)
        event_f1 = f1_score(y_event_val, event_preds)
        
        # 2. Time prediction metrics
        time_c_index = concordance_index(y_time_val, time_preds)
        
        # 3. Action prediction metrics
        action_accuracy = accuracy_score(y_action_val, action_preds)
        action_f1 = f1_score(y_action_val, action_preds, average='weighted')
        
        # Print validation metrics
        print("\nValidation Metrics:")
        print("-------------------")
        print(f"Event Prediction - AUC: {event_auc:.4f}, Precision: {event_precision:.4f}, Recall: {event_recall:.4f}, F1: {event_f1:.4f}")
        print(f"Time Prediction - Concordance Index: {time_c_index:.4f}")
        print(f"Action Prediction - Accuracy: {action_accuracy:.4f}, F1 (weighted): {action_f1:.4f}")
        
        # Store validation metrics
        self.validation_metrics = {
            'event': {
                'auc': event_auc,
                'precision': event_precision,
                'recall': event_recall,
                'f1': event_f1
            },
            'time': {
                'c_index': time_c_index
            },
            'action': {
                'accuracy': action_accuracy,
                'f1': action_f1
            }
        }
    
    def predict(self, X):
        """
        Generate predictions for all three tasks.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
            
        Returns:
        --------
        dict
            Dictionary containing predictions for all three tasks
        """
        if any(model is None for model in self.models.values()):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Preprocess features
        X_processed = self._preprocess_features(X, is_training=False)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_processed)
        
        # Generate predictions
        event_probs = self.models['event'].predict(dmatrix)
        event_preds = (event_probs > 0.5).astype(int)
        
        time_preds = self.models['time'].predict(dmatrix)
        
        action_probs = self.models['action'].predict(dmatrix)
        action_preds = np.argmax(action_probs, axis=1)
        action_labels = self.label_encoder.inverse_transform(action_preds)
        
        # Return predictions
        return {
            'event_probability': event_probs,
            'event_prediction': event_preds,
            'time_to_event': time_preds,
            'action_probabilities': action_probs,
            'action_prediction': action_labels
        }
    
    def evaluate(self, X, y_event, y_time, y_action):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test features
        y_event : pandas.Series
            Test binary target
        y_time : pandas.Series
            Test time-to-event target
        y_action : pandas.Series
            Test CHW action target
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics for all three tasks
        """
        # Preprocess features
        X_processed = self._preprocess_features(X, is_training=False)
        
        # Encode action labels
        y_action_encoded = self.label_encoder.transform(y_action)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_processed)
        
        # Generate predictions
        event_probs = self.models['event'].predict(dmatrix)
        event_preds = (event_probs > 0.5).astype(int)
        
        time_preds = self.models['time'].predict(dmatrix)
        
        action_probs = self.models['action'].predict(dmatrix)
        action_preds = np.argmax(action_probs, axis=1)
        
        # Calculate metrics
        # 1. Event prediction metrics
        event_auc = roc_auc_score(y_event, event_probs)
        event_precision = precision_score(y_event, event_preds)
        event_recall = recall_score(y_event, event_preds)
        event_f1 = f1_score(y_event, event_preds)
        
        # 2. Time prediction metrics
        time_c_index = concordance_index(y_time, time_preds)
        
        # 3. Action prediction metrics
        action_accuracy = accuracy_score(y_action_encoded, action_preds)
        action_f1 = f1_score(y_action_encoded, action_preds, average='weighted')
        action_report = classification_report(y_action_encoded, action_preds, 
                                             target_names=self.action_classes, 
                                             output_dict=True)
        
        # Compile metrics
        metrics = {
            'event': {
                'auc': event_auc,
                'precision': event_precision,
                'recall': event_recall,
                'f1': event_f1
            },
            'time': {
                'c_index': time_c_index
            },
            'action': {
                'accuracy': action_accuracy,
                'f1': action_f1,
                'report': action_report
            }
        }
        
        return metrics
    
    def get_feature_importance(self, plot=True, save_path=None):
        """
        Get feature importance for all three models.
        
        Parameters:
        -----------
        plot : bool
            Whether to plot feature importance
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        dict
            Dictionary containing feature importance for all three models
        """
        if any(model is None for model in self.models.values()):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        importance = {}
        
        # Get feature importance for each model
        for model_name, model in self.models.items():
            # Skip time model as it uses a different objective
            if model_name == 'time':
                continue
                
            importance[model_name] = model.get_score(importance_type='gain')
            
            # Normalize importance
            total = sum(importance[model_name].values())
            importance[model_name] = {k: v / total for k, v in importance[model_name].items()}
        
        # Plot feature importance
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Event model
            event_imp = pd.Series(importance['event']).sort_values(ascending=False).head(15)
            sns.barplot(x=event_imp.values, y=event_imp.index, ax=axes[0])
            axes[0].set_title('Event Prediction Feature Importance')
            axes[0].set_xlabel('Normalized Importance (Gain)')
            
            # Action model
            action_imp = pd.Series(importance['action']).sort_values(ascending=False).head(15)
            sns.barplot(x=action_imp.values, y=action_imp.index, ax=axes[1])
            axes[1].set_title('CHW Action Recommendation Feature Importance')
            axes[1].set_xlabel('Normalized Importance (Gain)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature importance plot saved to {save_path}")
            
            plt.show()
        
        return importance
    
    def generate_shap_values(self, X, task='event', plot=True, save_path=None):
        """
        Generate SHAP values for model interpretability.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        task : str
            Which task to generate SHAP values for ('event', 'time', or 'action')
        plot : bool
            Whether to plot SHAP values
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        numpy.ndarray
            SHAP values
        """
        if self.models[task] is None:
            raise ValueError(f"Model for {task} has not been trained yet. Call fit() first.")
        
        # Preprocess features
        X_processed = self._preprocess_features(X, is_training=False)
        
        # Create explainer
        explainer = shap.TreeExplainer(self.models[task])
        
        # Generate SHAP values
        shap_values = explainer(X_processed)
        
        # Plot SHAP values
        if plot:
            plt.figure(figsize=(12, 8))
            
            if task == 'action':
                # For multi-class, we need to specify which class to plot
                shap.summary_plot(shap_values[:, :, 0], X_processed, plot_type='bar')
            else:
                shap.summary_plot(shap_values, X_processed)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP values plot saved to {save_path}")
            
            plt.show()
        
        return shap_values
    
    def save_model(self, directory='models'):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        directory : str
            Directory to save the model
        """
        if any(model is None for model in self.models.values()):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each model component
        for model_name, model in self.models.items():
            model.save_model(os.path.join(directory, f'imt_xgb_{model_name}_model.json'))
        
        # Save metadata
        metadata = {
            'missingness_features': self.missingness_features,
            'event_params': self.event_params,
            'time_params': self.time_params,
            'action_params': self.action_params,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'action_classes': self.action_classes.tolist() if self.action_classes is not None else None,
            'validation_metrics': self.validation_metrics if hasattr(self, 'validation_metrics') else None,
            '_medians': self._medians if hasattr(self, '_medians') else None,
            '_most_frequent': self._most_frequent if hasattr(self, '_most_frequent') else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(directory, 'imt_xgb_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {directory}")
    
    @classmethod
    def load_model(cls, directory='models'):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        directory : str
            Directory containing the saved model
            
        Returns:
        --------
        IMT_XGBoost
            Loaded model
        """
        # Load metadata
        with open(os.path.join(directory, 'imt_xgb_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create model instance
        model = cls(
            missingness_features=metadata['missingness_features'],
            event_params=metadata['event_params'],
            time_params=metadata['time_params'],
            action_params=metadata['action_params'],
            random_state=metadata['random_state']
        )
        
        # Load model components
        model.models = {}
        for model_name in ['event', 'time', 'action']:
            model.models[model_name] = xgb.Booster()
            model.models[model_name].load_model(os.path.join(directory, f'imt_xgb_{model_name}_model.json'))
        
        # Restore metadata
        model.feature_names = metadata['feature_names']
        
        if metadata['action_classes'] is not None:
            model.label_encoder = LabelEncoder()
            model.label_encoder.classes_ = np.array(metadata['action_classes'])
            model.action_classes = np.array(metadata['action_classes'])
        
        if metadata.get('_medians') is not None:
            model._medians = metadata['_medians']
            
        if metadata.get('_most_frequent') is not None:
            model._most_frequent = metadata['_most_frequent']
            
        if metadata.get('validation_metrics') is not None:
            model.validation_metrics = metadata['validation_metrics']
        
        print(f"Model loaded from {directory}")
        print(f"Model timestamp: {metadata['timestamp']}")
        
        return model


def compare_with_baselines(X, y_event, y_time, y_action, random_state=42):
    """
    Compare IMT-XGB with baseline models.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features
    y_event : pandas.Series
        Binary target for event occurrence prediction
    y_time : pandas.Series
        Time-to-event target for survival analysis
    y_action : pandas.Series
        Categorical target for CHW action recommendation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from lifelines import WeibullAFTFitter
    
    # Split data
    X_train, X_test, ye_train, ye_test, yt_train, yt_test, ya_train, ya_test = train_test_split(
        X, y_event, y_time, y_action, test_size=0.2, random_state=random_state, stratify=y_event
    )
    
    # Encode action labels
    label_encoder = LabelEncoder()
    ya_train_encoded = label_encoder.fit_transform(ya_train)
    ya_test_encoded = label_encoder.transform(ya_test)
    
    # Initialize results dictionary
    results = {
        'event': {},
        'time': {},
        'action': {}
    }
    
    # 1. Baseline models for event prediction
    print("Training baseline models for event prediction...")
    
    # Logistic Regression
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train, ye_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_preds = lr.predict(X_test)
    
    results['event']['logistic_regression'] = {
        'auc': roc_auc_score(ye_test, lr_probs),
        'precision': precision_score(ye_test, lr_preds),
        'recall': recall_score(ye_test, lr_preds),
        'f1': f1_score(ye_test, lr_preds)
    }
    
    # Random Forest
    rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf.fit(X_train, ye_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    
    results['event']['random_forest'] = {
        'auc': roc_auc_score(ye_test, rf_probs),
        'precision': precision_score(ye_test, rf_preds),
        'recall': recall_score(ye_test, rf_preds),
        'f1': f1_score(ye_test, rf_preds)
    }
    
    # Standard XGBoost
    xgb_model = xgb.XGBClassifier(random_state=random_state)
    xgb_model.fit(X_train, ye_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    
    results['event']['xgboost'] = {
        'auc': roc_auc_score(ye_test, xgb_probs),
        'precision': precision_score(ye_test, xgb_preds),
        'recall': recall_score(ye_test, xgb_preds),
        'f1': f1_score(ye_test, xgb_preds)
    }
    
    # 2. Baseline models for time prediction
    print("Training baseline models for time prediction...")
    
    # Weibull AFT model from lifelines
    aft = WeibullAFTFitter()
    
    # Prepare data for lifelines
    df_surv = X_train.copy()
    df_surv['time'] = yt_train
    df_surv['event'] = ye_train
    
    # Fit model
    aft.fit(df_surv, duration_col='time', event_col='event')
    
    # Predict
    aft_preds = aft.predict_median(X_test)
    
    # Calculate concordance index
    results['time']['weibull_aft'] = {
        'c_index': concordance_index(yt_test, aft_preds)
    }
    
    # 3. Baseline models for action prediction
    print("Training baseline models for CHW action recommendation...")
    
    # Random Forest
    rf_action = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf_action.fit(X_train, ya_train_encoded)
    rf_action_preds = rf_action.predict(X_test)
    
    results['action']['random_forest'] = {
        'accuracy': accuracy_score(ya_test_encoded, rf_action_preds),
        'f1': f1_score(ya_test_encoded, rf_action_preds, average='weighted')
    }
    
    # Standard XGBoost
    xgb_action = xgb.XGBClassifier(random_state=random_state, objective='multi:softprob', num_class=len(label_encoder.classes_))
    xgb_action.fit(X_train, ya_train_encoded)
    xgb_action_preds = xgb_action.predict(X_test)
    
    results['action']['xgboost'] = {
        'accuracy': accuracy_score(ya_test_encoded, xgb_action_preds),
        'f1': f1_score(ya_test_encoded, xgb_action_preds, average='weighted')
    }
    
    # 4. Train IMT-XGB
    print("Training IMT-XGB model...")
    imt_xgb = IMT_XGBoost(random_state=random_state)
    imt_xgb.fit(X_train, ye_train, yt_train, ya_train)
    
    # Evaluate IMT-XGB
    imt_metrics = imt_xgb.evaluate(X_test, ye_test, yt_test, ya_test)
    
    # Add IMT-XGB results
    results['event']['imt_xgb'] = imt_metrics['event']
    results['time']['imt_xgb'] = imt_metrics['time']
    results['action']['imt_xgb'] = {
        'accuracy': imt_metrics['action']['accuracy'],
        'f1': imt_metrics['action']['f1']
    }
    
    return results, imt_xgb


if __name__ == "__main__":
    # Load synthetic data
    data_path = 'data/synthetic_medicaid_data.csv'
    df = pd.read_csv(data_path)
    
    # Prepare features and targets
    X = df.drop(['patient_id', 'event_occurred', 'time_to_event', 'chw_action'], axis=1)
    y_event = df['event_occurred']
    y_time = df['time_to_event']
    y_action = df['chw_action']
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Compare IMT-XGB with baseline models
    print("Comparing IMT-XGB with baseline models...")
    comparison_results, imt_model = compare_with_baselines(X, y_event, y_time, y_action)
    
    # Save comparison results
    with open('results/model_comparison.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)
    
    # Save IMT-XGB model
    imt_model.save_model('models')
    
    # Generate feature importance plot
    imt_model.get_feature_importance(save_path='figures/feature_importance.png')
    
    # Generate SHAP values for interpretability
    X_sample = X.sample(100, random_state=42)  # Use a sample for SHAP analysis
    imt_model.generate_shap_values(X_sample, task='event', save_path='figures/shap_values_event.png')
    imt_model.generate_shap_values(X_sample, task='action', save_path='figures/shap_values_action.png')
    
    print("IMT-XGB model training and evaluation complete!")
