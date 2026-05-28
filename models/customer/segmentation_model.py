import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import logging

logger = logging.getLogger(__name__)

class CustomerSegmentationModel:
    """Advanced customer segmentation using HDBSCAN and UMAP"""
    
    def __init__(self):
        self.umap_reducer = None
        self.hdbscan_clusterer = None
        self.kmeans_model = None
        self.segment_labels = None
        self.is_fitted = False
        
    def perform_umap_reduction(self, X: np.ndarray, 
                               n_components: int = 2,
                               n_neighbors: int = 15,
                               min_dist: float = 0.1) -> np.ndarray:
        """Reduce dimensionality using UMAP"""
        self.umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        X_reduced = self.umap_reducer.fit_transform(X)
        logger.info(f"UMAP reduction complete. Shape: {X_reduced.shape}")
        
        return X_reduced
    
    def perform_hdbscan_clustering(self, X: np.ndarray,
                                   min_cluster_size: int = 30,
                                   min_samples: int = 10) -> np.ndarray:
        """Perform HDBSCAN clustering"""
        self.hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = self.hdbscan_clusterer.fit_predict(X)
        
        # Calculate clustering quality
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
        else:
            silhouette = 0
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        logger.info(f"Silhouette score: {silhouette:.3f}")
        
        self.is_fitted = True
        
        return labels
    
    def perform_kmeans_segmentation(self, X: np.ndarray, 
                                   n_clusters: int = 5) -> Tuple[np.ndarray, float]:
        """Perform K-means segmentation"""
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = self.kmeans_model.fit_predict(X)
        inertia = self.kmeans_model.inertia_
        
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0
        
        logger.info(f"K-means complete. Inertia: {inertia:.2f}, Silhouette: {silhouette:.3f}")
        
        return labels, silhouette
    
    def find_optimal_clusters(self, X: np.ndarray, 
                             max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouettes = []
        
        for k in range(2, min(max_clusters + 1, len(X))):
            labels, _ = self.perform_kmeans_segmentation(X, n_clusters=k)
            inertias.append(self.kmeans_model.inertia_)
            
            if k > 1:
                silhouettes.append(silhouette_score(X, labels))
        
        # Elbow detection
        if len(inertias) > 1:
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            optimal_k = np.argmin(delta_deltas) + 2
        else:
            optimal_k = 2
        
        return optimal_k
    
    def profile_segments(self, df: pd.DataFrame, 
                        labels: np.ndarray,
                        feature_cols: List[str]) -> pd.DataFrame:
        """Profile and describe each segment"""
        df = df.copy()
        df['Segment'] = labels
        
        profiles = []
        
        for segment in sorted(df['Segment'].unique()):
            if segment == -1:
                segment_name = 'Noise/Outliers'
            else:
                segment_name = f'Segment {segment}'
            
            segment_data = df[df['Segment'] == segment]
            
            profile = {
                'Segment': segment_name,
                'Size': len(segment_data),
                'Percentage': len(segment_data) / len(df) * 100
            }
            
            # Feature statistics
            for col in feature_cols:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    profile[f'{col}_mean'] = segment_data[col].mean()
                    profile[f'{col}_median'] = segment_data[col].median()
                    profile[f'{col}_std'] = segment_data[col].std()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def assign_segment_names(self, profiles_df: pd.DataFrame,
                            feature_cols: List[str]) -> Dict[int, str]:
        """Assign meaningful names to segments based on their characteristics"""
        segment_names = {}
        
        for _, row in profiles_df.iterrows():
            segment_id = row['Segment']
            
            # Determine characteristics
            high_value = False
            frequent = False
            recent = False
            
            if 'TotalSpend_mean' in row.index:
                high_value = row['TotalSpend_mean'] > profiles_df['TotalSpend_mean'].median()
            
            if 'Frequency_mean' in row.index:
                frequent = row['Frequency_mean'] > profiles_df['Frequency_mean'].median()
            
            if 'RecencyDays_mean' in row.index:
                recent = row['RecencyDays_mean'] < profiles_df['RecencyDays_mean'].median()
            
            if high_value and frequent:
                if recent:
                    name = 'VIP Champions'
                else:
                    name = 'At-Risk VIPs'
            elif high_value:
                name = 'Big Spenders'
            elif frequent and recent:
                name = 'Loyal Regulars'
            elif recent:
                name = 'New Active'
            elif frequent:
                name = 'Loyal but Lapsing'
            else:
                name = 'At Risk'
            
            segment_names[segment_id] = name
        
        return segment_names