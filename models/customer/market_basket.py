import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MarketBasketAnalyzer:
    """Market basket analysis for cross-selling and product recommendations"""
    
    def __init__(self):
        self.frequent_itemsets = None
        self.rules = None
        self.product_lookup = {}
        self.is_fitted = False
        
    def find_frequent_itemsets(self, transactions: List[List[str]],
                              min_support: float = 0.01) -> pd.DataFrame:
        """Find frequent itemsets using Apriori algorithm"""
        all_items = sorted(set(item for transaction in transactions for item in transaction))
        self.product_lookup = {item: idx for idx, item in enumerate(all_items)}
        
        encoded = []
        for transaction in transactions:
            row = {item: 1 for item in transaction}
            encoded.append(row)
        
        encoded_df = pd.DataFrame(encoded).fillna(0).astype(bool)
        
        self.frequent_itemsets = apriori(
            encoded_df, 
            min_support=min_support, 
            use_colnames=True, 
            max_len=3
        )
        
        logger.info(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        self.is_fitted = True
        
        return self.frequent_itemsets
    
    def generate_rules(self, min_lift: float = 1.0,
                      min_confidence: float = 0.1) -> pd.DataFrame:
        """Generate association rules"""
        if self.frequent_itemsets is None:
            raise ValueError("Must find frequent itemsets first")
        
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="lift",
            min_threshold=min_lift
        )
        
        self.rules = self.rules[self.rules['confidence'] >= min_confidence]
        
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        logger.info(f"Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def get_recommendations(self, items: List[str], 
                           top_n: int = 5) -> List[Dict]:
        """Get product recommendations based on items"""
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        recommendations = []
        
        for item in items:
            item_rules = self.rules[
                self.rules['antecedents'].apply(lambda x: item in x)
            ]
            
            for _, rule in item_rules.iterrows():
                consequents = list(rule['consequents'])
                for consequent in consequents:
                    recommendations.append({
                        'input_item': item,
                        'recommended_item': consequent,
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'support': rule['support']
                    })
        
        recommendations_df = pd.DataFrame(recommendations)
        if len(recommendations_df) > 0:
            recommendations_df = recommendations_df.sort_values(
                'lift', ascending=False
            ).drop_duplicates(subset=['recommended_item'])
        
        return recommendations_df.head(top_n).to_dict('records')
    
    def calculate_product_affinity_matrix(self) -> pd.DataFrame:
        """Calculate product affinity matrix"""
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        all_products = set()
        for _, rule in self.rules.iterrows():
            all_products.update(rule['antecedents'])
            all_products.update(rule['consequents'])
        
        products_list = sorted(all_products)
        
        affinity = pd.DataFrame(
            index=products_list,
            columns=products_list,
            data=0.0
        )
        
        for _, rule in self.rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            for ant in antecedents:
                for cons in consequents:
                    affinity.loc[ant, cons] = rule['lift']
        
        return affinity
    
    def find_cross_sell_opportunities(self, product_categories: List[str],
                                     top_n: int = 10) -> pd.DataFrame:
        """Find cross-selling opportunities between categories"""
        if self.rules is None:
            raise ValueError("Must generate rules first")
        
        opportunities = []
        
        for i, cat1 in enumerate(product_categories):
            for cat2 in product_categories[i+1:]:
                # Find rules between these categories
                cross_rules = self.rules[
                    (self.rules['antecedents'].apply(lambda x: any(cat1 in str(item) for item in x))) &
                    (self.rules['consequents'].apply(lambda x: any(cat2 in str(item) for item in x)))
                ]
                
                if len(cross_rules) > 0:
                    avg_lift = cross_rules['lift'].mean()
                    max_lift = cross_rules['lift'].max()
                    total_rules = len(cross_rules)
                    
                    opportunities.append({
                        'Category1': cat1,
                        'Category2': cat2,
                        'AvgLift': avg_lift,
                        'MaxLift': max_lift,
                        'NumRules': total_rules,
                        'Strength': avg_lift * np.log1p(total_rules)
                    })
        
        opportunities_df = pd.DataFrame(opportunities)
        if len(opportunities_df) > 0:
            opportunities_df = opportunities_df.sort_values('Strength', ascending=False)
        
        return opportunities_df.head(top_n)
    
    def calculate_basket_metrics(self, transactions: List[List[str]]) -> Dict:
        """Calculate overall basket metrics"""
        basket_sizes = [len(t) for t in transactions]
        
        # Calculate frequency of items
        item_freq = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_freq[item] += 1
        
        # Calculate co-occurrence
        co_occurrence = defaultdict(lambda: defaultdict(int))
        for transaction in transactions:
            for i, item1 in enumerate(transaction):
                for item2 in transaction[i+1:]:
                    co_occurrence[item1][item2] += 1
                    co_occurrence[item2][item1] += 1
        
        return {
            'avg_basket_size': np.mean(basket_sizes),
            'median_basket_size': np.median(basket_sizes),
            'max_basket_size': max(basket_sizes),
            'total_unique_items': len(item_freq),
            'most_popular_item': max(item_freq, key=item_freq.get),
            'most_popular_count': max(item_freq.values()),
            'avg_item_frequency': np.mean(list(item_freq.values()))
        }