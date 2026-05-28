import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CustomerMetrics:
    """Customer-centric metrics and KPIs"""
    
    @staticmethod
    def calculate_customer_acquisition_cost(marketing_spend: float,
                                           new_customers: int) -> float:
        """Calculate Customer Acquisition Cost (CAC)"""
        return marketing_spend / new_customers if new_customers > 0 else float('inf')
    
    @staticmethod
    def calculate_customer_retention_rate(customers_start: int,
                                         customers_end: int,
                                         new_customers: int) -> float:
        """Calculate customer retention rate"""
        if customers_start == 0:
            return 0
        return (customers_end - new_customers) / customers_start * 100
    
    @staticmethod
    def calculate_net_revenue_retention(revenue_start: float,
                                       revenue_end: float,
                                       expansion_revenue: float,
                                       churned_revenue: float) -> float:
        """Calculate Net Revenue Retention (NRR)"""
        total_revenue = revenue_start + expansion_revenue - churned_revenue
        return total_revenue / revenue_start * 100 if revenue_start > 0 else 0
    
    @staticmethod
    def calculate_customer_health_score(recency: float, frequency: float,
                                       monetary: float, 
                                       weights: Dict[str, float] = None) -> float:
        """Calculate customer health score (0-100)"""
        if weights is None:
            weights = {'recency': 0.4, 'frequency': 0.3, 'monetary': 0.3}
        
        # Normalize inputs (assuming they're already on similar scales)
        score = (
            weights['recency'] * recency +
            weights['frequency'] * frequency +
            weights['monetary'] * monetary
        )
        
        return min(max(score, 0), 100)
    
    @staticmethod
    def calculate_cohort_metrics(df: pd.DataFrame, 
                                cohort_col: str,
                                date_col: str,
                                value_col: str,
                                periods: int = 12) -> pd.DataFrame:
        """Calculate cohort retention metrics"""
        # Create cohort groups
        df['Cohort'] = df[cohort_col].dt.to_period('M')
        df['Period'] = (df[date_col].dt.to_period('M') - df['Cohort']).apply(lambda x: x.n)
        
        # Calculate cohort metrics
        cohort_data = df.groupby(['Cohort', 'Period']).agg({
            value_col: 'sum',
            'CustomerID': 'nunique'
        }).reset_index()
        
        # Pivot for cohort table
        cohort_pivot = cohort_data.pivot(
            index='Cohort',
            columns='Period',
            values=value_col
        )
        
        # Calculate retention from initial period
        cohort_retention = cohort_pivot.div(cohort_pivot[0], axis=0) * 100
        
        return cohort_retention
    
    @staticmethod
    def calculate_customer_migration_matrix(df: pd.DataFrame,
                                           current_period_col: str,
                                           previous_period_col: str) -> pd.DataFrame:
        """Calculate customer segment migration matrix"""
        migration = pd.crosstab(
            df[previous_period_col],
            df[current_period_col],
            normalize='index'
        ) * 100
        
        return migration
    
    @staticmethod
    def calculate_share_of_wallet(customer_spend: float,
                                 category_total_spend: float) -> float:
        """Calculate share of wallet"""
        return customer_spend / category_total_spend * 100 if category_total_spend > 0 else 0
    
    @staticmethod
    def calculate_customer_concentration(total_customers: int,
                                        top_n_customers: int,
                                        top_n_revenue: float,
                                        total_revenue: float) -> Dict:
        """Calculate customer concentration metrics"""
        return {
            'revenue_concentration': top_n_revenue / total_revenue * 100,
            'customer_concentration': top_n_customers / total_customers * 100,
            'herfindahl_index': (top_n_revenue / total_revenue) ** 2 * 10000
        }