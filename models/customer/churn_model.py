import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    """Advanced churn prediction using survival analysis and machine learning"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20,
            random_state=42, class_weight='balanced'
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        )
        self.cox_model = None
        self.kmf = KaplanMeierFitter()
        self.is_trained = False
        self.feature_importance = None
        
    def fit_survival_model(self, df: pd.DataFrame, duration_col: str = 'Time',
                          event_col: str = 'Churned', 
                          feature_cols: Optional[list] = None) -> Dict:
        """Fit Cox Proportional Hazards model"""
        try:
            if feature_cols is None:
                exclude_cols = ['CustomerID', 'LastOrderDate', 'Churned', 'Time', 
                              'FirstOrder', 'LastOrder']
                feature_cols = [col for col in df.columns 
                              if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
            
            model_data = df[[duration_col, event_col] + feature_cols].dropna()
            
            self.cox_model = CoxPHFitter()
            self.cox_model.fit(
                model_data, 
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False
            )
            
            summary = self.cox_model.summary
            
            c_index = self.cox_model.concordance_index_
            
            self.is_trained = True
            
            return {
                'concordance_index': c_index,
                'summary': summary,
                'significant_features': summary[summary['p'] < 0.05].index.tolist(),
                'hazard_ratios': summary['exp(coef)'].to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to fit survival model: {e}")
            raise
    
    def fit_random_forest(self, X: np.ndarray, y: np.ndarray, 
                         feature_names: list) -> Dict:
        """Fit Random Forest classifier for churn prediction"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.rf_model.predict(X_test)
            y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'accuracy': self.rf_model.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importance': self.feature_importance
            }
            
            self.is_trained = True
            
            os.makedirs('models/customer', exist_ok=True)
            joblib.dump(self.rf_model, 'models/customer/churn_rf_model.pkl')
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fit Random Forest: {e}")
            raise
    
    def predict_churn_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict churn probability"""
        if not self.is_trained:
            if os.path.exists('models/customer/churn_rf_model.pkl'):
                self.rf_model = joblib.load('models/customer/churn_rf_model.pkl')
                self.is_trained = True
            else:
                raise ValueError("Model not trained yet")
        
        return self.rf_model.predict_proba(X)[:, 1]
    
    def calculate_churn_risk_score(self, churn_prob: float, 
                                   customer_value: float,
                                   recency_days: float) -> float:
        """Calculate composite churn risk score (0-100)"""
        # Weighted combination of churn probability, customer value, and recency
        value_factor = np.log1p(customer_value) / np.log1p(100000)  # Normalize
        
        risk_score = (
            0.5 * churn_prob * 100 +
            0.3 * (1 / (1 + np.exp(-recency_days / 180))) * 100 +
            0.2 * value_factor * 100
        )
        
        return min(risk_score, 100)
    
    def get_risk_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Segment customers by churn risk"""
        df = df.copy()
        
        if 'ChurnProbability' in df.columns:
            conditions = [
                (df['ChurnProbability'] >= 0.7),
                (df['ChurnProbability'] >= 0.3) & (df['ChurnProbability'] < 0.7),
                (df['ChurnProbability'] < 0.3)
            ]
            choices = ['High Risk', 'Medium Risk', 'Low Risk']
            df['RiskSegment'] = np.select(conditions, choices, default='Unknown')
        
        return df
    
    def estimate_retention_impact(self, churn_prob: float, 
                                  clv: float,
                                  retention_cost: float = 100) -> Dict:
        """Estimate the financial impact of retention efforts"""
        # Expected loss if customer churns
        expected_loss = churn_prob * clv
        
        # ROI of retention
        if retention_cost > 0:
            retention_roi = (expected_loss - retention_cost) / retention_cost * 100
        else:
            retention_roi = float('inf')
        
        should_retain = expected_loss > retention_cost
        
        return {
            'expected_loss': expected_loss,
            'retention_roi': retention_roi,
            'should_retain': should_retain,
            'break_even_probability': retention_cost / clv if clv > 0 else 1
        }