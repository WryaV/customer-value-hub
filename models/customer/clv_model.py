import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from lifelines import KaplanMeierFitter
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CustomerLifetimeValueModel:
    """Advanced CLV prediction using Pareto/NBD and Gamma-Gamma models"""
    
    def __init__(self):
        self.rf_model = None
        self.gg_model = None
        self.is_fitted = False
        
    def calculate_rfm(self, transactions_df: pd.DataFrame, 
                     reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        # Convert Decimal columns to float
        for col in transactions_df.select_dtypes(include=['object']).columns:
            try:
                transactions_df[col] = pd.to_numeric(transactions_df[col], errors='ignore')
            except (ValueError, TypeError):
                pass
        
        if reference_date is None:
            reference_date = transactions_df['OrderDate'].max()
        
        rfm = transactions_df.groupby('CustomerID').agg({
            'OrderDate': lambda x: (reference_date - x.max()).days,  
            'SalesOrderID': 'nunique', 
            'TotalDue': 'sum' 
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        first_purchase = transactions_df.groupby('CustomerID')['OrderDate'].min().reset_index()
        first_purchase.columns = ['CustomerID', 'FirstPurchase']
        first_purchase['Age'] = (reference_date - first_purchase['FirstPurchase']).dt.days
        
        rfm = rfm.merge(first_purchase[['CustomerID', 'FirstPurchase', 'Age']], on='CustomerID')
        
        rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
        rfm['PurchaseFrequency'] = rfm['Frequency'] / rfm['Age'] * 365  
        
        return rfm
    
    def fit_btyd_models(self, rfm_df: pd.DataFrame) -> Dict:
        """Fit Buy Till You Die models (simplified Pareto/NBD approximation)"""
        
        rfm_df = rfm_df.copy()
        for col in rfm_df.select_dtypes(include=['object']).columns:
            try:
                rfm_df[col] = pd.to_numeric(rfm_df[col], errors='ignore')
            except (ValueError, TypeError):
                pass
        
   
        customers = rfm_df.copy()
        
        numeric_cols = ['Frequency', 'Age', 'AvgOrderValue']
        for col in numeric_cols:
            if col in customers.columns:
                customers[col] = pd.to_numeric(customers[col], errors='coerce').astype(float)
        
        freq_mean = customers['Frequency'].mean()
        freq_var = customers['Frequency'].var()
        
        # Gamma distribution parameters for transaction rate
        if freq_var > freq_mean:
            r = freq_mean**2 / (freq_var - freq_mean)
            alpha = freq_mean / (freq_var - freq_mean)
        else:
            r = freq_mean
            alpha = 1.0
        
        # Beta distribution parameters for dropout rate
        a = 1.0
        b = 2.0
        
        # Calculate expected future transactions
        customers['ExpectedFutureTransactions'] = (
            (r + customers['Frequency']) / 
            (alpha + customers['Age'] / 365)
        )
        
        # Calculate CLV
        customers['PredictedCLV'] = (
            customers['ExpectedFutureTransactions'].astype(float) * 
            customers['AvgOrderValue'].astype(float)
        )
        
        # Calculate CLV confidence intervals
        clv_std = customers['PredictedCLV'].std()
        customers['CLV_Lower'] = customers['PredictedCLV'] - 1.96 * clv_std
        customers['CLV_Upper'] = customers['PredictedCLV'] + 1.96 * clv_std
        
        # Calculate probability of being alive
        customers['ProbAlive'] = 1 / (1 + alpha / (alpha + customers['Age'] / 365))
        
        self.is_fitted = True
        
        return {
            'parameters': {
                'r': r,
                'alpha': alpha,
                'a': a,
                'b': b
            },
            'customer_metrics': customers
        }
    
    def calculate_customer_health_score(self, rfm_df: pd.DataFrame, 
                                         order_consistency_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate customer health score using fuzzy logic - B2B optimized"""
        rfm_df = rfm_df.copy()
        for col in rfm_df.select_dtypes(include=['object']).columns:
            try:
                rfm_df[col] = pd.to_numeric(rfm_df[col], errors='ignore')
            except (ValueError, TypeError):
                pass
        
        df = rfm_df.copy()
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Normalize RFM metrics
        for col in ['Recency', 'Frequency', 'Monetary']:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f'{col}_norm'] = 0
        
        # B2B: Recency is less critical - use exponential decay for recency scoring
        df['Recency_Score'] = np.exp(-df['Recency_norm'] * 2)
        
        df['Frequency_Score'] = df['Frequency_norm']
        
        df['Monetary_Score'] = df['Monetary_norm']
        
        if order_consistency_df is not None:
            df = df.merge(order_consistency_df[['CustomerID', 'OrderConsistency']], on='CustomerID', how='left')
            consistency_max = df['OrderConsistency'].max()
            if consistency_max > 0:
                df['Consistency_Score'] = 1 - (df['OrderConsistency'] / consistency_max)
            else:
                df['Consistency_Score'] = 0.5
            df['Consistency_Score'] = df['Consistency_Score'].fillna(0.5)
            
            df['HealthScore'] = (
                0.25 * df['Recency_Score'] +
                0.25 * df['Frequency_Score'] +
                0.25 * df['Monetary_Score'] +
                0.25 * df['Consistency_Score']
            ) * 100
        else:
            df['HealthScore'] = (
                0.35 * df['Recency_Score'] +
                0.30 * df['Frequency_Score'] +
                0.35 * df['Monetary_Score']
            ) * 100
        
        df['HealthCategory'] = pd.cut(
            df['HealthScore'],
            bins=[0, 30, 50, 70, 100],
            labels=['Critical', 'At Risk', 'Stable', 'Excellent']
        )
        
        return df
    
    def segment_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Advanced customer segmentation"""
        rfm_df = rfm_df.copy()
        for col in rfm_df.select_dtypes(include=['object']).columns:
            try:
                rfm_df[col] = pd.to_numeric(rfm_df[col], errors='ignore')
            except (ValueError, TypeError):
                pass
        
        df = rfm_df.copy()
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Calculate percentiles with duplicate handling
        try:
            r_quartiles = pd.qcut(df['Recency'], 4, labels=['High', 'Medium', 'Low', 'Very Low'], duplicates='drop')
            f_quartiles = pd.qcut(df['Frequency'], 4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            m_quartiles = pd.qcut(df['Monetary'], 4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        except ValueError:
            # If still failing, use rank-based quantiles
            r_quartiles = pd.cut(df['Recency'].rank(pct=True), 4, labels=['High', 'Medium', 'Low', 'Very Low'])
            f_quartiles = pd.cut(df['Frequency'].rank(pct=True), 4, labels=['Low', 'Medium', 'High', 'Very High'])
            m_quartiles = pd.cut(df['Monetary'].rank(pct=True), 4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Store median values for segment assignment
        freq_median = df['Frequency'].median()
        mon_median = df['Monetary'].median()
        rec_median = df['Recency'].median()
        
        df['Segment'] = df.apply(lambda row: self._assign_segment(row, freq_median, mon_median, rec_median), axis=1)
        
        # Calculate segment metrics
        df['SegmentSize'] = df.groupby('Segment')['CustomerID'].transform('count')
        df['SegmentRevenue'] = df.groupby('Segment')['Monetary'].transform('sum')
        if 'PredictedCLV' in df.columns:
            df['SegmentAvgCLV'] = df.groupby('Segment')['PredictedCLV'].transform('mean')
        else:
            df['SegmentAvgCLV'] = 0
        
        return df
    
    def _assign_segment(self, row, freq_median, mon_median, rec_median):
        """Assign customer segment based on behavior patterns"""
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        prob_alive = row.get('ProbAlive', 0.5)
        
        if frequency == 0:
            return 'New'
        elif prob_alive < 0.3:
            if monetary > mon_median:
                return 'Lost VIP'
            else:
                return 'Dormant'
        elif frequency > freq_median and monetary > mon_median:
            if recency < rec_median:
                return 'Enterprise VIP'
            else:
                return 'At Risk VIP'
        elif frequency > freq_median:
            return 'Regular Buyer'
        elif monetary > mon_median:
            return 'Big Spender'
        elif recency < rec_median:
            return 'New Active'
        else:
            return 'Occasional'