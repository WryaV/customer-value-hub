import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class CustomerPlotlyComponents:
    """Reusable Plotly visualization components for customer analytics"""
    
    @staticmethod
    def create_rfm_cube(rfm_df: pd.DataFrame, 
                       color_col: str = 'Segment',
                       title: str = "RFM Customer Cube") -> go.Figure:
        """Create 3D RFM scatter plot"""
        fig = go.Figure(data=[go.Scatter3d(
            x=rfm_df['Recency'],
            y=rfm_df['Frequency'],
            z=rfm_df['Monetary'],
            mode='markers',
            marker=dict(
                size=rfm_df['Monetary'] / rfm_df['Monetary'].max() * 20 + 5,
                color=rfm_df[color_col].astype('category').cat.codes if color_col in rfm_df.columns else rfm_df['Recency'],
                colorscale='Viridis',
                opacity=0.7,
                showscale=True,
                colorbar=dict(title=color_col)
            ),
            text=[f"Customer {id}<br>R: {r}<br>F: {f}<br>M: ${m:,.0f}" 
                  for id, r, f, m in zip(rfm_df['CustomerID'], 
                                         rfm_df['Recency'], 
                                         rfm_df['Frequency'], 
                                         rfm_df['Monetary'])],
            hovertemplate='%{text}'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Recency (days)',
                yaxis_title='Frequency',
                zaxis_title='Monetary ($)'
            ),
            height=700
        )
        
        return fig
    
    @staticmethod
    def create_chord_diagram(source_indices: List[int], target_indices: List[int],
                            values: List[float], labels: List[str],
                            title: str = "Product Affinity Chord") -> go.Figure:
        """Create chord diagram for relationships"""
        nodes = []
        for label in labels:
            nodes.append(dict(name=label))
        
        links = []
        for s, t, v in zip(source_indices, target_indices, values):
            links.append(dict(source=s, target=t, value=v))
        
        n_colors = len(labels)
        colors = px.colors.qualitative.Set3[:n_colors]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=[colors[s] for s in source_indices]
            )
        )])
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_clv_distribution(clv_values: List[float],
                               title: str = "Customer Lifetime Value Distribution") -> go.Figure:
        """Create CLV distribution histogram with fitted curve"""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Distribution', 'Cumulative'),
                           row_heights=[0.6, 0.4])
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=clv_values,
                nbinsx=50,
                name='CLV Distribution',
                histnorm='probability density',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # KDE curve
        from scipy import stats
        kde_x = np.linspace(min(clv_values), max(clv_values), 100)
        kde = stats.gaussian_kde(clv_values)
        kde_y = kde(kde_x)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color='darkblue', width=2)
            ),
            row=1, col=1
        )
        
        # Add percentile lines
        p25 = np.percentile(clv_values, 25)
        p50 = np.percentile(clv_values, 50)
        p75 = np.percentile(clv_values, 75)
        p90 = np.percentile(clv_values, 90)
        
        for p, label in [(p25, '25th'), (p50, '50th'), (p75, '75th'), (p90, '90th')]:
            fig.add_vline(x=p, line_dash="dash", line_color="red", 
                         annotation_text=label, row=1, col=1)
        
        # Cumulative distribution
        sorted_clv = np.sort(clv_values)
        cumulative = np.arange(1, len(sorted_clv) + 1) / len(sorted_clv)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_clv,
                y=cumulative,
                mode='lines',
                name='Cumulative',
                fill='tozeroy',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=700,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_markov_transition_heatmap(transition_matrix: pd.DataFrame,
                                        title: str = "Customer State Transitions") -> go.Figure:
        """Create Markov transition matrix heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=transition_matrix.columns,
            y=transition_matrix.index,
            colorscale='Blues',
            text=np.round(transition_matrix.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='To State',
            yaxis_title='From State',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_streamgraph(df: pd.DataFrame, x_col: str, y_col: str,
                          group_col: str, 
                          title: str = "Revenue Streams") -> go.Figure:
        """Create streamgraph for showing composition over time"""
        fig = go.Figure()
        
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            
            fig.add_trace(go.Scatter(
                x=group_data[x_col],
                y=group_data[y_col],
                mode='lines',
                stackgroup='one',
                name=str(group),
                line=dict(width=0.5)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_geo_density_map(latitudes: List[float], longitudes: List[float],
                              values: List[float] = None,
                              title: str = "Customer Geographic Distribution") -> go.Figure:
        """Create geographic density map of customers"""
        fig = go.Figure()
        
        if values is None:
            values = [1] * len(latitudes)
        
        fig.add_trace(go.Densitymapbox(
            lat=latitudes,
            lon=longitudes,
            z=values,
            radius=10,
            colorscale='Viridis',
            showscale=True,
            hovertemplate='Lat: %{lat}<br>Lon: %{lon}<br>Value: %{z}'
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=np.mean(latitudes), lon=np.mean(longitudes)),
                zoom=3
            ),
            title=title,
            height=600
        )
        
        return fig