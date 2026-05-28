import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
import logging
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)

class CustomerFeatureEngineer:
    """Feature engineering for customer analytics"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def engineer_customer_features(self, sales_df: pd.DataFrame, 
                                   customers_df: pd.DataFrame,
                                   demographics_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create advanced customer-level features"""
        
        # Aggregate sales by customer
        customer_sales = sales_df.groupby('CustomerID').agg({
            'SalesOrderID': 'nunique',
            'OrderDate': ['min', 'max', 'std'],
            'TotalDue': ['sum', 'mean', 'std', 'min', 'max'],
            'OrderQty': ['sum', 'mean'],
            'UnitPriceDiscount': 'mean',
            'ProductID': 'nunique'
        }).reset_index()
        
        # Flatten column names
        customer_sales.columns = ['CustomerID', 'NumOrders', 'FirstOrder', 'LastOrder', 
                                  'OrderDateStd', 'TotalSpend', 'AvgOrderValue', 
                                  'SpendStd', 'MinOrder', 'MaxOrder', 
                                  'TotalItems', 'AvgItemsPerOrder',
                                  'AvgDiscount', 'UniqueProducts']
        
        # Temporal features
        customer_sales['FirstOrder'] = pd.to_datetime(customer_sales['FirstOrder'])
        customer_sales['LastOrder'] = pd.to_datetime(customer_sales['LastOrder'])
        
        reference_date = customer_sales['LastOrder'].max()
        customer_sales['RecencyDays'] = (reference_date - customer_sales['LastOrder']).dt.days
        customer_sales['CustomerLifetimeDays'] = (customer_sales['LastOrder'] - customer_sales['FirstOrder']).dt.days
        customer_sales['CustomerLifetimeDays'] = customer_sales['CustomerLifetimeDays'].clip(lower=1)
        
        # Behavioral metrics
        customer_sales['PurchaseFrequency'] = customer_sales['NumOrders'] / customer_sales['CustomerLifetimeDays'] * 30
        customer_sales['AvgDaysBetweenOrders'] = customer_sales['CustomerLifetimeDays'] / customer_sales['NumOrders']
        customer_sales['SpendVelocity'] = customer_sales['TotalSpend'] / customer_sales['CustomerLifetimeDays'] * 30
        
        # B2B: Order consistency (standard deviation of order intervals)
        def calculate_order_consistency(cust_id):
            cust_orders = sales_df[sales_df['CustomerID'] == cust_id].sort_values('OrderDate')
            if len(cust_orders) >= 3:
                intervals = cust_orders['OrderDate'].diff().dt.days.dropna()
                return intervals.std() if len(intervals) > 0 else 0
            return 0
        
        consistency_scores = []
        for cust_id in customer_sales['CustomerID'].unique():
            consistency = calculate_order_consistency(cust_id)
            consistency_scores.append({'CustomerID': cust_id, 'OrderConsistency': consistency})
        
        consistency_df = pd.DataFrame(consistency_scores)
        customer_sales = customer_sales.merge(consistency_df, on='CustomerID', how='left')
        customer_sales['OrderConsistency'] = customer_sales['OrderConsistency'].fillna(0)
        
        # B2B: Volume trend (increasing/decreasing over time)
        trends = []
        for cust_id in customer_sales['CustomerID'].unique():
            cust_orders = sales_df[sales_df['CustomerID'] == cust_id].sort_values('OrderDate')
            if len(cust_orders) >= 3:
                x = np.arange(len(cust_orders))
                y = cust_orders['OrderQty'].values
                slope, _, _, _ = stats.theilslopes(y, x)
                trends.append({'CustomerID': cust_id, 'VolumeTrend': slope})
            else:
                trends.append({'CustomerID': cust_id, 'VolumeTrend': 0})
        
        trend_df = pd.DataFrame(trends)
        customer_sales = customer_sales.merge(trend_df, on='CustomerID', how='left')
        
        # Consistency metrics
        customer_sales['CV_Spend'] = customer_sales['SpendStd'] / customer_sales['AvgOrderValue']
        customer_sales['CV_Spend'] = customer_sales['CV_Spend'].fillna(0)
        
        # Discount dependency
        customer_sales['DiscountDependency'] = customer_sales['AvgDiscount'] * 100
        customer_sales['HighDiscountBuyer'] = (customer_sales['AvgDiscount'] > 0.1).astype(int)
        
        # Product diversity (Shannon entropy)
        product_mix = sales_df.groupby(['CustomerID', 'CategoryName'])['OrderQty'].sum().reset_index()
        product_pivot = product_mix.pivot(index='CustomerID', columns='CategoryName', values='OrderQty').fillna(0)
        
        entropies = []
        for cust_id in product_pivot.index:
            probs = product_pivot.loc[cust_id].values
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append({'CustomerID': cust_id, 'ProductEntropy': entropy})
        
        entropy_df = pd.DataFrame(entropies)
        customer_sales = customer_sales.merge(entropy_df, on='CustomerID', how='left')
        
        # B2B: Product category penetration
        category_counts = sales_df.groupby('CustomerID')['CategoryName'].nunique().reset_index()
        category_counts.columns = ['CustomerID', 'CategoryPenetration']
        customer_sales = customer_sales.merge(category_counts, on='CustomerID', how='left')
        
        # Customer type features
        if customers_df is not None:
            customer_sales = customer_sales.merge(
                customers_df[['CustomerID', 'TerritoryID', 'PersonID', 'StoreID']],
                on='CustomerID', how='left'
            )
            customer_sales['IsIndividual'] = customer_sales['PersonID'].notna().astype(int)
            customer_sales['IsStore'] = customer_sales['StoreID'].notna().astype(int)
        
        # Demographic features
        if demographics_df is not None:
            customer_sales = customer_sales.merge(
                demographics_df, left_on='PersonID', right_on='BusinessEntityID', how='left'
            )
            
            if 'YearlyIncome' in customer_sales.columns:
                income_map = {
                    '0-30000': 15000, '30001-60000': 45000,
                    '60001-100000': 80000, '100001-150000': 125000,
                    '150001+': 200000
                }
                customer_sales['IncomeNumeric'] = customer_sales['YearlyIncome'].map(income_map)
        
        return customer_sales
    
    def prepare_market_basket_data(self, sales_df: pd.DataFrame, 
                                   min_products: int = 2) -> List[List[str]]:
        """Prepare data for market basket analysis"""
        # Filter orders with at least min_products
        order_product_counts = sales_df.groupby('SalesOrderID')['ProductID'].count()
        valid_orders = order_product_counts[order_product_counts >= min_products].index
        
        # Create transactions
        transactions = []
        for order_id in valid_orders:
            order_products = sales_df[sales_df['SalesOrderID'] == order_id]['ProductName'].tolist()
            transactions.append(order_products)
        
        return transactions
    
    def calculate_market_basket_metrics(self, transactions: List[List[str]], 
                                       min_support: float = 0.01,
                                       min_lift: float = 1.0) -> pd.DataFrame:
        """Calculate market basket analysis metrics"""
        from mlxtend.frequent_patterns import apriori, association_rules
        
        # Create one-hot encoded matrix
        all_products = sorted(set(item for transaction in transactions for item in transaction))
        
        # For large product sets, limit to top products
        if len(all_products) > 100:
            product_counts = pd.Series([item for t in transactions for item in t]).value_counts()
            all_products = product_counts.head(100).index.tolist()
        
        encoded = []
        for transaction in transactions:
            row = {product: (product in transaction) for product in all_products}
            encoded.append(row)
        
        encoded_df = pd.DataFrame(encoded)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True, max_len=3)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
            rules = rules.sort_values('lift', ascending=False)
            return rules
        
        return pd.DataFrame()
    
    def calculate_behavioral_segmentation_features(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced behavioral segmentation features"""
        customer_features = sales_df.groupby('CustomerID').agg({
            'SalesOrderID': 'nunique',
            'TotalDue': ['sum', 'mean', 'std'],
            'OrderQty': 'sum',
            'UnitPriceDiscount': ['mean', 'std'],
            'OrderDate': lambda x: (x.max() - x.min()).days
        }).reset_index()
        
        # Flatten columns
        customer_features.columns = ['CustomerID', 'Frequency', 'Monetary_Total', 
                                     'Monetary_Avg', 'Monetary_Std', 'TotalQty',
                                     'Discount_Avg', 'Discount_Std', 'LifetimeDays']
        
        # Calculate derived metrics
        customer_features['Monetary_CV'] = customer_features['Monetary_Std'] / customer_features['Monetary_Avg']
        customer_features['ItemsPerOrder'] = customer_features['TotalQty'] / customer_features['Frequency']
        
        # Weekend vs weekday ratio
        sales_df['OrderDate'] = pd.to_datetime(sales_df['OrderDate'])
        sales_df['DayOfWeek'] = sales_df['OrderDate'].dt.dayofweek
        sales_df['IsWeekend'] = sales_df['DayOfWeek'].isin([5, 6]).astype(int)
        
        weekend_ratio = sales_df.groupby('CustomerID').agg({
            'IsWeekend': 'mean',
            'SalesOrderID': 'nunique'
        }).reset_index()
        weekend_ratio['WeekendRatio'] = weekend_ratio['IsWeekend']
        
        customer_features = customer_features.merge(
            weekend_ratio[['CustomerID', 'WeekendRatio']], 
            on='CustomerID', how='left'
        )
        
        # Order value trend (using Theil-Sen estimator)
        trends = []
        for cust_id in customer_features['CustomerID'].unique():
            cust_orders = sales_df[sales_df['CustomerID'] == cust_id].sort_values('OrderDate')
            if len(cust_orders) >= 3:
                x = np.arange(len(cust_orders))
                y = cust_orders['TotalDue'].values
                slope, _, _, _ = stats.theilslopes(y, x)
                trends.append({'CustomerID': cust_id, 'SpendTrend': slope})
            else:
                trends.append({'CustomerID': cust_id, 'SpendTrend': 0})
        
        trend_df = pd.DataFrame(trends)
        customer_features = customer_features.merge(trend_df, on='CustomerID', how='left')
        
        return customer_features
    
    def create_survival_features(self, sales_df: pd.DataFrame, 
                                 churn_window_days: int = 365) -> pd.DataFrame:
        """Create features for survival analysis - B2B focused (365+ days for churn)"""
        customer_last_order = sales_df.groupby('CustomerID')['OrderDate'].max().reset_index()
        customer_last_order.columns = ['CustomerID', 'LastOrderDate']
        
        reference_date = pd.to_datetime(sales_df['OrderDate'].max())
        customer_last_order['DaysSinceLastOrder'] = (
            reference_date - pd.to_datetime(customer_last_order['LastOrderDate'])
        ).dt.days
        
        # Define churn event
        customer_last_order['Churned'] = (
            customer_last_order['DaysSinceLastOrder'] > churn_window_days
        ).astype(int)
        
        # Also add high churn flag for 24+ months
        customer_last_order['HighChurn'] = (
            customer_last_order['DaysSinceLastOrder'] > 730
        ).astype(int)
        
        # Time to event
        customer_last_order['Time'] = customer_last_order['DaysSinceLastOrder']
        
        # Add customer features
        customer_features = self.engineer_customer_features(sales_df, None)
        customer_last_order = customer_last_order.merge(
            customer_features, on='CustomerID', how='left'
        )
        
        return customer_last_order
    
    def calculate_granger_causality_features(self, sales_df: pd.DataFrame,
                                            category_col: str = 'CategoryName',
                                            date_col: str = 'OrderDate',
                                            max_lag: int = 4) -> pd.DataFrame:
        """Calculate Granger causality between product categories"""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        sales_df[date_col] = pd.to_datetime(sales_df[date_col])
        
        # Create weekly sales by category
        weekly_sales = sales_df.groupby([
            pd.Grouper(key=date_col, freq='W'), category_col
        ])['TotalDue'].sum().reset_index()
        
        # Pivot to get time series for each category
        pivot_sales = weekly_sales.pivot(
            index=date_col, columns=category_col, values='TotalDue'
        ).fillna(0)
        
        # Calculate Granger causality
        categories = pivot_sales.columns
        granger_matrix = pd.DataFrame(index=categories, columns=categories)
        granger_pvalues = pd.DataFrame(index=categories, columns=categories)
        
        for cause in categories:
            for effect in categories:
                if cause != effect:
                    test_data = pivot_sales[[effect, cause]].dropna()
                    if len(test_data) > max_lag + 5:
                        try:
                            gc_result = grangercausalitytests(
                                test_data, maxlag=max_lag, verbose=False
                            )
                            # Use minimum p-value across all lags
                            min_pvalue = min(
                                gc_result[lag][0]['ssr_chi2test'][1] 
                                for lag in range(1, max_lag + 1)
                            )
                            granger_matrix.loc[cause, effect] = 1 / (min_pvalue + 1e-10)
                            granger_pvalues.loc[cause, effect] = min_pvalue
                        except:
                            granger_matrix.loc[cause, effect] = 0
                            granger_pvalues.loc[cause, effect] = 1
        
        return granger_matrix.astype(float), granger_pvalues.astype(float)