import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class CustomerCustomCharts:
    """Custom advanced chart components for customer analytics"""
    
    @staticmethod
    def create_uplift_curve(uplift_df: pd.DataFrame,
                           title: str = "Uplift Curve") -> go.Figure:
        """Create uplift curve for promotion effectiveness"""
        fig = go.Figure()
        
        # Treatment group
        fig.add_trace(go.Scatter(
            x=uplift_df['population_fraction'],
            y=uplift_df['treatment_response'],
            mode='lines',
            name='Treatment',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=uplift_df['population_fraction'],
            y=uplift_df['control_response'],
            mode='lines',
            name='Control',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=uplift_df['population_fraction'],
            y=uplift_df['uplift'],
            mode='lines',
            name='Uplift',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Population Fraction',
            yaxis_title='Response',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_bass_diffusion_curve(time_points: np.ndarray,
                                   adoption: np.ndarray,
                                   fitted_adoption: np.ndarray,
                                   p_coef: float, q_coef: float,
                                   title: str = "Product Adoption (Bass Model)") -> go.Figure:
        """Create Bass diffusion model visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=adoption,
            mode='markers',
            name='Actual Adoption',
            marker=dict(size=8, color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=fitted_adoption,
            mode='lines',
            name='Bass Model Fit',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"{title}<br><sup>p (innovation): {p_coef:.4f}, q (imitation): {q_coef:.4f}, q/p: {q_coef/p_coef:.2f}</sup>",
            xaxis_title='Time',
            yaxis_title='Cumulative Adoption',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_parallel_coordinates(df: pd.DataFrame, 
                                   dimensions: List[str],
                                   color_col: Optional[str] = None,
                                   title: str = "Customer Parallel Coordinates") -> go.Figure:
        """Create parallel coordinates plot for multi-dimensional analysis"""
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[color_col].values if color_col else df[dimensions[0]].values,
                colorscale='Viridis',
                showscale=True,
                cmin=df[color_col].min() if color_col else df[dimensions[0]].min(),
                cmax=df[color_col].max() if color_col else df[dimensions[0]].max()
            ),
            dimensions=[
                dict(
                    label=dim,
                    values=df[dim].values,
                    range=[df[dim].min(), df[dim].max()]
                ) for dim in dimensions
            ]
        ))
        
        fig.update_layout(
            title=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_waterfall_chart(categories: List[str], values: List[float],
                              title: str = "Waterfall Analysis") -> go.Figure:
        """Create waterfall chart"""
        fig = go.Figure(go.Waterfall(
            name="Waterfall",
            orientation="v",
            measure=["relative"] * len(categories),
            x=categories,
            y=values,
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_customer_journey_sankey(states: List[str],
                                      transitions: List[Tuple[int, int, float]],
                                      title: str = "Customer Journey Flow") -> go.Figure:
        """Create customer journey Sankey diagram"""
        source_indices = [t[0] for t in transitions]
        target_indices = [t[1] for t in transitions]
        values = [t[2] for t in transitions]
        
        # Node colors based on state
        colors = []
        for state in states:
            if 'VIP' in state or 'Champion' in state:
                colors.append('gold')
            elif 'Risk' in state or 'Churn' in state:
                colors.append('red')
            elif 'Active' in state or 'Loyal' in state:
                colors.append('green')
            elif 'New' in state:
                colors.append('lightblue')
            else:
                colors.append('gray')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=states,
                color=colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=[f"rgba({np.random.randint(100,200)},{np.random.randint(100,200)},{np.random.randint(100,200)},0.4)" 
                      for _ in range(len(source_indices))]
            )
        )])
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_cohort_heatmap(cohort_matrix: pd.DataFrame,
                             title: str = "Customer Cohort Retention") -> go.Figure:
        """Create cohort retention heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=cohort_matrix.values,
            x=cohort_matrix.columns,
            y=cohort_matrix.index,
            colorscale='RdYlGn',
            text=np.round(cohort_matrix.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title='Retention %')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Period',
            yaxis_title='Cohort',
            height=500
        )
        
        return fig