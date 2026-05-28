import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StatisticalTests:
    """Collection of statistical tests for customer analytics"""
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray = None) -> Dict:
        """Perform chi-square test"""
        if expected is None:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        else:
            chi2, p_value = stats.chisquare(observed, expected)
            dof = len(observed) - 1
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'expected': expected
        }
    
    @staticmethod
    def t_test_independent(group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Perform independent t-test"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / s_pooled if s_pooled > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    @staticmethod
    def anova_test(*groups) -> Dict:
        """Perform one-way ANOVA"""
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Eta squared (effect size)
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((v - grand_mean)**2 for v in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'eta_squared': eta_squared,
            'effect_size': 'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'
        }
    
    @staticmethod
    def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix with significance"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
        p_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
        
        for i in numeric_cols:
            for j in numeric_cols:
                if method == 'pearson':
                    corr, p_val = stats.pearsonr(df[i].dropna(), df[j].dropna())
                else:
                    corr, p_val = stats.spearmanr(df[i].dropna(), df[j].dropna())
                
                corr_matrix.loc[i, j] = corr
                p_matrix.loc[i, j] = p_val
        
        return corr_matrix.astype(float), p_matrix.astype(float)
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, 
                                     statistic: callable = np.mean,
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Dict:
        """Calculate bootstrap confidence interval"""
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        return {
            'statistic': statistic(data),
            'ci_lower': lower,
            'ci_upper': upper,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_level': confidence
        }