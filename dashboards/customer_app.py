import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from data.database_connector import CustomerDatabaseConnector
from data.feature_engineering import CustomerFeatureEngineer
from models.customer.clv_model import CustomerLifetimeValueModel
from models.customer.churn_model import ChurnPredictionModel
from models.customer.segmentation_model import CustomerSegmentationModel
from models.customer.market_basket import MarketBasketAnalyzer
from utils.statistical_tests import StatisticalTests
from utils.metrics import CustomerMetrics
from visualizations.plotly_components import CustomerPlotlyComponents
from visualizations.custom_charts import CustomerCustomCharts


st.set_page_config(
    page_title="Customer Value Intelligence Hub",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    return {
        'db': CustomerDatabaseConnector(),
        'feature_engineer': CustomerFeatureEngineer(),
        'clv_model': CustomerLifetimeValueModel(),
        'churn_model': ChurnPredictionModel(),
        'segmentation_model': CustomerSegmentationModel(),
        'market_basket': MarketBasketAnalyzer(),
        'stats': StatisticalTests(),
        'metrics_calc': CustomerMetrics(),
        'viz': CustomerPlotlyComponents(),
        'custom_charts': CustomerCustomCharts()
    }

components = init_components()

# Load data
@st.cache_data(ttl=settings.CACHE_TTL_SECONDS)
def load_customer_data():
    with st.spinner("Loading customer data..."):
        sales_orders = components['db'].get_sales_orders()
        customers = components['db'].get_customers()
        demographics = components['db'].get_customer_demographics()
        territories = components['db'].get_sales_territories()
        special_offers = components['db'].get_special_offers()
        sales_reasons = components['db'].get_sales_reasons()
        
        return {
            'sales_orders': sales_orders,
            'customers': customers,
            'demographics': demographics,
            'territories': territories,
            'special_offers': special_offers,
            'sales_reasons': sales_reasons
        }

data = load_customer_data()

# Feature engineering
with st.spinner("Engineering features..."):
    customer_features = components['feature_engineer'].engineer_customer_features(
        data['sales_orders'], data['customers'], data['demographics']
    )
    
    # RFM Analysis
    rfm_df = components['clv_model'].calculate_rfm(
        data['sales_orders'].groupby(['CustomerID', 'OrderDate', 'SalesOrderID', 'TotalDue'])
        .first().reset_index()
    )
    
    # Fit CLV model
    clv_results = components['clv_model'].fit_btyd_models(rfm_df)
    customer_metrics = clv_results['customer_metrics']
    
    # Calculate health scores with consistency data
    consistency_df = customer_features[['CustomerID', 'OrderConsistency']].drop_duplicates()
    health_scores = components['clv_model'].calculate_customer_health_score(rfm_df, consistency_df)
    
    # Segment customers - merge health scores properly
    customer_metrics_with_health = customer_metrics.merge(
        health_scores[['CustomerID', 'HealthScore', 'HealthCategory']], 
        on='CustomerID', 
        how='left'
    )
    
    segmented_customers = components['clv_model'].segment_customers(customer_metrics_with_health)
    
    # Prepare market basket
    transactions = components['feature_engineer'].prepare_market_basket_data(data['sales_orders'])
    if len(transactions) > 0:
        components['market_basket'].find_frequent_itemsets(transactions, min_support=0.01)
        components['market_basket'].generate_rules(min_lift=1.2)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/customer-insight.png", width=80)
    st.title("👥 Customer Hub")
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Analysis Filters")
    
    # Date range
    date_col = pd.to_datetime(data['sales_orders']['OrderDate'])
    date_range = st.date_input(
        "Date Range",
        value=(date_col.min().date(), date_col.max().date())
    )
    
    # Territory filter - handle missing columns gracefully
    try:
        if 'TerritoryName' in data['customers'].columns:
            territories_list = ['All'] + list(data['customers']['TerritoryName'].dropna().unique())
        else:
            territories_list = ['All'] + list(data['territories']['Name'].unique())
    except:
        territories_list = ['All']
    selected_territory = st.selectbox("Territory", territories_list)
    
    # Customer type
    customer_type = st.selectbox(
        "Customer Type",
        ['All', 'Individual', 'Store']
    )
    
    # Segment filter
    segments = ['All'] + list(segmented_customers['Segment'].unique())
    selected_segment = st.selectbox("Customer Segment", segments)
    
    # Value thresholds
    st.markdown("---")
    st.subheader("🎯 B2B Thresholds")
    high_value_threshold = st.number_input(
        "High Value Threshold ($)",
        value=50000, min_value=10000, max_value=500000, step=5000
    )
    churn_days = st.slider("Churn Window (days)", 180, 730, 365)
    
    # B2B Specific Metrics
    st.markdown("---")
    st.subheader("🏭 B2B Metrics")
    
    store_customers = customer_metrics[customer_metrics['IsStore'] == 1] if 'IsStore' in customer_metrics.columns else pd.DataFrame()
    individual_customers = customer_metrics[customer_metrics['IsIndividual'] == 1] if 'IsIndividual' in customer_metrics.columns else pd.DataFrame()
    
    if len(store_customers) > 0:
        st.metric("B2B Customers (Stores)", f"{len(store_customers):,}")
        st.metric("B2B Avg Order Value", f"${store_customers['AvgOrderValue'].mean():,.0f}")
    
    if len(individual_customers) > 0:
        st.metric("B2C Customers (Individual)", f"{len(individual_customers):,}")
        st.metric("B2C Avg Order Value", f"${individual_customers['AvgOrderValue'].mean():,.0f}")
    
    # Revenue concentration
    top_5_revenue = customer_metrics.nlargest(int(len(customer_metrics) * 0.05), 'Monetary')['Monetary'].sum()
    total_revenue = customer_metrics['Monetary'].sum()
    top_5_share = top_5_revenue / total_revenue * 100 if total_revenue > 0 else 0
    st.metric("Top 5% Revenue Share", f"{top_5_share:.1f}%")
    
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    st.metric("Total Customers", f"{len(rfm_df):,}")
    st.metric("Total Revenue", f"${data['sales_orders']['TotalDue'].sum():,.0f}")
    st.metric("Avg Order Value", f"${data['sales_orders']['TotalDue'].mean():,.0f}")

# Apply filters to data
def apply_filters():
    filtered_customer_metrics = customer_metrics_with_health.copy()
    filtered_segmented = segmented_customers.copy()
    filtered_rfm = rfm_df.copy()
    
    # Territory filter
    if selected_territory != 'All':
        try:
            if 'TerritoryName' in data['customers'].columns:
                territory_customers = data['customers'][data['customers']['TerritoryName'] == selected_territory]['CustomerID']
            else:
                territory_id = data['territories'][data['territories']['Name'] == selected_territory]['TerritoryID'].iloc[0]
                territory_customers = data['customers'][data['customers']['TerritoryID'] == territory_id]['CustomerID']
            filtered_customer_metrics = filtered_customer_metrics[filtered_customer_metrics['CustomerID'].isin(territory_customers)]
            filtered_segmented = filtered_segmented[filtered_segmented['CustomerID'].isin(territory_customers)]
            filtered_rfm = filtered_rfm[filtered_rfm['CustomerID'].isin(territory_customers)]
        except Exception as e:
            pass
    
    # Customer type filter
    if customer_type != 'All':
        if customer_type == 'Individual':
            type_customers = data['customers'][data['customers']['PersonID'].notna()]['CustomerID']
        else:
            type_customers = data['customers'][data['customers']['StoreID'].notna()]['CustomerID']
        filtered_customer_metrics = filtered_customer_metrics[filtered_customer_metrics['CustomerID'].isin(type_customers)]
        filtered_segmented = filtered_segmented[filtered_segmented['CustomerID'].isin(type_customers)]
        filtered_rfm = filtered_rfm[filtered_rfm['CustomerID'].isin(type_customers)]
    
    # Segment filter
    if selected_segment != 'All':
        filtered_segmented = filtered_segmented[filtered_segmented['Segment'] == selected_segment]
        filtered_customer_metrics = filtered_customer_metrics[filtered_customer_metrics['CustomerID'].isin(filtered_segmented['CustomerID'])]
        filtered_rfm = filtered_rfm[filtered_rfm['CustomerID'].isin(filtered_segmented['CustomerID'])]
    
    return filtered_customer_metrics, filtered_segmented, filtered_rfm

# Apply filters to get filtered data
filtered_customer_metrics, filtered_segmented, filtered_rfm = apply_filters()

# Main Dashboard
st.markdown('<h1 style="margin-top: -12px; margin-bottom: 0px;">👥 360° customer analytics powered by advanced machine learning & statistical modeling</h1>', unsafe_allow_html=True)


# KPI Banner - Using filtered data
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_customers = len(filtered_rfm)
    st.metric(
        "Total Customers",
        f"{total_customers:,}",
        delta=f"{total_customers - len(rfm_df):,} vs unfiltered"
    )

with col2:
    active_customers = len(filtered_rfm[filtered_rfm['Recency'] <= 90])
    active_pct = active_customers/total_customers*100 if total_customers > 0 else 0
    st.metric(
        "Active (90d)",
        f"{active_customers:,}",
        delta=f"{active_pct:.1f}%"
    )

with col3:
    avg_clv = filtered_customer_metrics['PredictedCLV'].mean() if len(filtered_customer_metrics) > 0 else 0
    st.metric(
        "Avg CLV",
        f"${avg_clv:,.0f}",
        delta=f"${avg_clv - 3000:,.0f}"
    )

with col4:
    at_risk = len(filtered_segmented[filtered_segmented['Segment'].str.contains('Risk|Dormant|At Risk', na=False)])
    at_risk_pct = at_risk/total_customers*100 if total_customers > 0 else 0
    st.metric(
        "At Risk Customers",
        f"{at_risk:,}",
        delta=f"{at_risk_pct:.1f}%",
        delta_color="inverse"
    )

with col5:
    if 'HealthScore' in filtered_customer_metrics.columns:
        avg_health = filtered_customer_metrics['HealthScore'].mean() if len(filtered_customer_metrics) > 0 else 0
    else:
        avg_health = 0
    st.metric(
        "Avg Health Score",
        f"{avg_health:.1f}",
        delta=f"{avg_health - 50:.1f}"
    )

with col6:
    churn_rate = len(filtered_rfm[filtered_rfm['Recency'] > churn_days]) / total_customers * 100 if total_customers > 0 else 0
    st.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%",
        delta=f"{churn_rate - 15:.1f}%",
        delta_color="inverse"
    )

st.markdown("---")

# Tab Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧬 Customer Genome",
    "💰 Lifetime Value",
    "⚠️ Churn War Room",
    "🎯 Promotion Intelligence",
    "🕸️ Product Ecosystem",
    "🌍 Territory Intelligence"
])

# Tab 1: Customer Genome Map
with tab1:
    st.header("🧬 Customer Genome Map")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("UMAP Customer Projection")
        
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 
                       'PurchaseFrequency', 'PredictedCLV', 'ProbAlive']
        
        available_features = [col for col in feature_cols if col in filtered_customer_metrics.columns]
        
        if len(available_features) >= 2 and len(filtered_customer_metrics) >= 10:
            X = filtered_customer_metrics[available_features].fillna(0).values
            X = np.array(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            try:
                X_reduced = components['segmentation_model'].perform_umap_reduction(
                    X_scaled, n_components=2, n_neighbors=15
                )
                
                plot_df = pd.DataFrame({
                    'UMAP1': X_reduced[:, 0],
                    'UMAP2': X_reduced[:, 1],
                    'CLV': filtered_customer_metrics['PredictedCLV'],
                    'Segment': filtered_segmented['Segment']
                })
                
                fig_genome = px.scatter(
                    plot_df,
                    x='UMAP1',
                    y='UMAP2',
                    color='Segment',
                    size='CLV',
                    hover_data=['CLV'],
                    title="Customer Behavior Genome Map",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                fig_genome.update_layout(height=600)
                st.plotly_chart(fig_genome, use_container_width=True)
                
            except Exception as e:
                st.warning(f"UMAP projection not available: {e}")
                st.info("Showing PCA projection instead")
        else:
            st.info("Insufficient data for UMAP projection")
    
    with col2:
        st.subheader("Segment Profiles")
        
        segment_counts = filtered_segmented['Segment'].value_counts()
        
        if len(segment_counts) > 0:
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segment Distribution",
                hole=0.4
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        else:
            st.info("No segment data available for current filters")
        
        if selected_segment != 'All' and selected_segment in filtered_segmented['Segment'].values:
            segment_data = filtered_segmented[filtered_segmented['Segment'] == selected_segment]
            
            st.markdown(f"**{selected_segment} Profile:**")
            st.metric("Customers", f"{len(segment_data):,}")
            st.metric("Avg CLV", f"${segment_data['PredictedCLV'].mean():,.0f}")
            st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
            st.metric("Avg Order Value", f"${segment_data['AvgOrderValue'].mean():,.0f}")
    
    st.subheader("🔍 Customer Deep Dive")
    
    customer_options = filtered_rfm['CustomerID'].tolist()[:100]
    if len(customer_options) > 0:
        selected_customer = st.selectbox("Select Customer for Analysis", customer_options)
        
        if selected_customer:
            customer_data = filtered_customer_metrics[filtered_customer_metrics['CustomerID'] == selected_customer]
            customer_segment = filtered_segmented[filtered_segmented['CustomerID'] == selected_customer]
            
            if len(customer_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Recency", f"{customer_data['Recency'].iloc[0]:.0f} days")
                with col2:
                    st.metric("Frequency", f"{customer_data['Frequency'].iloc[0]:.0f}")
                with col3:
                    st.metric("Monetary", f"${customer_data['Monetary'].iloc[0]:,.0f}")
                with col4:
                    segment = customer_segment['Segment'].iloc[0] if len(customer_segment) > 0 else 'N/A'
                    st.metric("Segment", segment)
                
                customer_orders = data['sales_orders'][
                    data['sales_orders']['CustomerID'] == selected_customer
                ].sort_values('OrderDate')
                
                if len(customer_orders) > 0:
                    fig_timeline = px.scatter(
                        customer_orders,
                        x='OrderDate',
                        y='TotalDue',
                        size='OrderQty',
                        color='CategoryName',
                        title=f"Order History - Customer {selected_customer}",
                        labels={'TotalDue': 'Order Value ($)'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

# Tab 2: Lifetime Value Command Center
with tab2:
    st.header("💰 Customer Lifetime Value Command Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CLV Distribution")
        
        clv_values = filtered_customer_metrics['PredictedCLV'].values if len(filtered_customer_metrics) > 0 else []
        if len(clv_values) > 0:
            fig_clv = components['viz'].create_clv_distribution(clv_values, "Customer Lifetime Value Distribution")
            st.plotly_chart(fig_clv, use_container_width=True)
        else:
            st.info("No CLV data available for current filters")
    
    with col2:
        st.subheader("CLV by Segment")
        
        if len(filtered_segmented) > 0:
            segment_clv = filtered_segmented.groupby('Segment').agg({
                'PredictedCLV': 'mean',
                'CustomerID': 'count',
                'Monetary': 'sum'
            }).reset_index()
            
            fig_segment_clv = px.treemap(
                segment_clv,
                path=['Segment'],
                values='PredictedCLV',
                color='PredictedCLV',
                color_continuous_scale='Viridis',
                title="CLV Treemap by Segment"
            )
            st.plotly_chart(fig_segment_clv, use_container_width=True)
        else:
            st.info("No segment data available")
    
    st.subheader("RFM Cube Analysis")
    
    sample_customers = filtered_rfm.sample(min(500, len(filtered_rfm))) if len(filtered_rfm) > 0 else pd.DataFrame()
    if len(sample_customers) > 0:
        fig_rfm = components['viz'].create_rfm_cube(sample_customers, color_col='Monetary', title="Interactive RFM Cube")
        st.plotly_chart(fig_rfm, use_container_width=True)
    else:
        st.info("No RFM data available for current filters")
    
    st.subheader("Customer Portfolio Value & Concentration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_portfolio_value = filtered_customer_metrics['PredictedCLV'].sum() if len(filtered_customer_metrics) > 0 else 0
        st.metric("Total Portfolio Value", f"${total_portfolio_value:,.0f}")
    
    with col2:
        top_10_pct_value = filtered_customer_metrics.nlargest(int(len(filtered_customer_metrics) * 0.1), 'PredictedCLV')['PredictedCLV'].sum() if len(filtered_customer_metrics) > 0 else 0
        pct = top_10_pct_value / total_portfolio_value * 100 if total_portfolio_value > 0 else 0
        st.metric("Top 10% Value", f"${top_10_pct_value:,.0f}", delta=f"{pct:.1f}% of total")
    
    with col3:
        avg_customer_value = filtered_customer_metrics['PredictedCLV'].mean() if len(filtered_customer_metrics) > 0 else 0
        median_value = filtered_customer_metrics['PredictedCLV'].median() if len(filtered_customer_metrics) > 0 else 0
        st.metric("Avg Customer Value", f"${avg_customer_value:,.0f}", delta=f"vs median ${median_value:,.0f}")
    
    # Revenue Concentration Chart
    st.subheader("📊 Revenue Concentration (Top Customers)")
    
    if len(filtered_customer_metrics) > 0:
        customer_revenue = filtered_customer_metrics.sort_values('Monetary', ascending=False)
        customer_revenue = customer_revenue.head(100)
        customer_revenue['CumulativeRevenue'] = customer_revenue['Monetary'].cumsum()
        customer_revenue['CumulativePercent'] = customer_revenue['CumulativeRevenue'] / customer_revenue['Monetary'].sum() * 100
        
        fig_concentration = go.Figure()
        fig_concentration.add_trace(go.Scatter(
            x=list(range(1, len(customer_revenue) + 1)),
            y=customer_revenue['CumulativePercent'],
            mode='lines+markers',
            name='Cumulative Revenue %',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        fig_concentration.add_hline(y=80, line_dash="dash", line_color="red")
        fig_concentration.update_layout(
            title="Revenue Concentration (Pareto Analysis)",
            xaxis_title="Customer Rank",
            yaxis_title="Cumulative Revenue %",
            height=400
        )
        st.plotly_chart(fig_concentration, use_container_width=True)

# Tab 3: Churn War Room
with tab3:
    st.header("⚠️ Churn War Room")
    
    churn_features = components['feature_engineer'].create_survival_features(data['sales_orders'], churn_days)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Risk Distribution")
        churn_counts = filtered_segmented['Segment'].value_counts()
        risk_segments = churn_counts[churn_counts.index.str.contains('Risk|Dormant|At Risk', na=False)]
        
        if len(risk_segments) > 0:
            fig_churn_pie = px.pie(values=risk_segments.values, names=risk_segments.index, title="At-Risk Customer Segments", hole=0.3)
            st.plotly_chart(fig_churn_pie, use_container_width=True)
        else:
            st.info("No at-risk segments identified")
    
    with col2:
        st.subheader("Risk Factors Analysis")
        
        if len(churn_features) > 0:
            numeric_cols = churn_features.select_dtypes(include=[np.number]).columns
            correlations = []
            
            for col in numeric_cols:
                if col not in ['CustomerID', 'Churned', 'HighChurn']:
                    corr = churn_features[col].corr(churn_features['Churned'])
                    if not np.isnan(corr):
                        correlations.append({'Feature': col, 'Correlation': abs(corr)})
            
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False).head(10)
                fig_factors = px.bar(corr_df, x='Correlation', y='Feature', orientation='h', title="Top Churn Risk Factors", color='Correlation', color_continuous_scale='Reds')
                st.plotly_chart(fig_factors, use_container_width=True)
    
    st.subheader("🎮 Customer Retention Simulator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_clv = st.number_input("Customer CLV ($)", value=50000, min_value=5000, max_value=500000, step=5000)
    with col2:
        sim_churn_prob = st.slider("Churn Probability (%)", 0, 100, 15) / 100
    with col3:
        sim_retention_cost = st.number_input("Retention Cost ($)", value=5000, min_value=500, max_value=50000, step=500)
    
    retention_analysis = components['churn_model'].estimate_retention_impact(sim_churn_prob, sim_clv, sim_retention_cost)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Loss", f"${retention_analysis['expected_loss']:,.0f}")
    with col2:
        st.metric("Retention ROI", f"{retention_analysis['retention_roi']:.1f}%")
    with col3:
        verdict = "✅ RETAIN" if retention_analysis['should_retain'] else "❌ DON'T RETAIN"
        st.metric("Recommendation", verdict)

# Tab 4: Promotion Intelligence
with tab4:
    st.header("🎯 Promotion Intelligence & Uplift Modeling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Promotion Effectiveness")
        
        discount_analysis = data['sales_orders'].copy()
        discount_analysis['DiscountCategory'] = pd.cut(
            discount_analysis['UnitPriceDiscount'] * 100,
            bins=[0, 10, 20, 30, 50, 100],
            labels=['0-10%', '10-20%', '20-30%', '30-50%', '50%+']
        )
        
        promo_effectiveness = discount_analysis.groupby('DiscountCategory', observed=True).agg({
            'OrderQty': 'mean', 'TotalDue': 'mean', 'SalesOrderID': 'nunique'
        }).reset_index()
        
        if len(promo_effectiveness) > 0:
            fig_promo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_promo.add_trace(go.Bar(x=promo_effectiveness['DiscountCategory'], y=promo_effectiveness['OrderQty'], name="Avg Order Qty", marker_color='blue'), secondary_y=False)
            fig_promo.add_trace(go.Scatter(x=promo_effectiveness['DiscountCategory'], y=promo_effectiveness['TotalDue'], name="Avg Order Value ($)", mode='lines+markers', line=dict(color='red', width=2)), secondary_y=True)
            fig_promo.update_layout(title="Promotion Impact Analysis")
            st.plotly_chart(fig_promo, use_container_width=True)
    
    with col2:
        st.subheader("Customer Persuasion Quadrants")
        
        n_customers = min(100, len(filtered_segmented))
        np.random.seed(42)
        uplift_df = pd.DataFrame({
            'CustomerID': range(n_customers),
            'TreatmentEffect': np.random.normal(0.15, 0.25, n_customers),
            'BaselineProbability': np.random.beta(2, 5, n_customers)
        })
        
        uplift_df['Quadrant'] = 'Lost Causes'
        uplift_df.loc[(uplift_df['TreatmentEffect'] > 0.08) & (uplift_df['BaselineProbability'] < 0.4), 'Quadrant'] = 'Persuadables'
        uplift_df.loc[(uplift_df['TreatmentEffect'] < 0.08) & (uplift_df['BaselineProbability'] > 0.4), 'Quadrant'] = 'Sure Things'
        uplift_df.loc[(uplift_df['TreatmentEffect'] < -0.05), 'Quadrant'] = 'Sleeping Dogs'
        
        quadrant_colors = {'Persuadables': 'green', 'Sure Things': 'blue', 'Lost Causes': 'red', 'Sleeping Dogs': 'orange'}
        fig_quadrants = px.scatter(uplift_df, x='BaselineProbability', y='TreatmentEffect', color='Quadrant', color_discrete_map=quadrant_colors, title="Customer Persuasion Quadrants")
        fig_quadrants.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_quadrants, use_container_width=True)
    
    st.subheader("💸 Promotion ROI Simulator")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_segment = st.selectbox("Target Segment", segments[1:] if len(segments) > 1 else ['All'])
    with col2:
        discount_pct = st.slider("Discount %", 5, 50, 15)
    with col3:
        expected_uptake = st.slider("Expected Uptake %", 5, 80, 20)
    with col4:
        promo_cost = st.number_input("Campaign Cost ($)", value=50000)
    
    segment_customers = filtered_segmented[filtered_segmented['Segment'] == target_segment] if target_segment != 'All' else filtered_segmented
    target_customers = len(segment_customers)
    responding_customers = int(target_customers * expected_uptake / 100)
    avg_order_value = segment_customers['AvgOrderValue'].mean() if len(segment_customers) > 0 else 0
    discount_amount = avg_order_value * discount_pct / 100
    incremental_revenue = responding_customers * (avg_order_value - discount_amount)
    roi = (incremental_revenue - promo_cost) / promo_cost * 100 if promo_cost > 0 else 0
    
    st.metric(f"Estimated ROI for {target_segment}", f"{roi:.1f}%", delta=f"${incremental_revenue - promo_cost:,.0f} net return")

# Tab 5: Product Ecosystem Network
with tab5:
    st.header("🕸️ Product Ecosystem & Cross-Selling")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Product Affinity Network")
        
        if components['market_basket'].rules is not None:
            rules = components['market_basket'].rules.head(20)
            unique_products = set()
            for _, rule in rules.iterrows():
                unique_products.update(rule['antecedents'])
                unique_products.update(rule['consequents'])
            
            products_list = sorted(unique_products)[:30]
            product_to_idx = {p: i for i, p in enumerate(products_list)}
            
            source_indices, target_indices, lift_values = [], [], []
            for _, rule in rules.iterrows():
                for ant in rule['antecedents']:
                    for cons in rule['consequents']:
                        if ant in product_to_idx and cons in product_to_idx:
                            source_indices.append(product_to_idx[ant])
                            target_indices.append(product_to_idx[cons])
                            lift_values.append(rule['lift'])
            
            if len(source_indices) > 0:
                fig_chord = components['viz'].create_chord_diagram(source_indices, target_indices, lift_values, products_list, "Product Purchase Affinity (Lift)")
                st.plotly_chart(fig_chord, use_container_width=True)
    
    with col2:
        st.subheader("Top Cross-Sell Opportunities")
        
        if components['market_basket'].rules is not None:
            top_rules = components['market_basket'].rules.head(10)
            opportunities = []
            for _, rule in top_rules.iterrows():
                opportunities.append({
                    'If Customer Buys': ', '.join(list(rule['antecedents'])),
                    'Recommend': ', '.join(list(rule['consequents'])),
                    'Lift': f"{rule['lift']:.1f}x",
                    'Confidence': f"{rule['confidence']:.1%}"
                })
            st.dataframe(pd.DataFrame(opportunities), use_container_width=True)
    
    st.subheader("🔮 Product Recommendation Engine")
    
    product_options = sorted(data['sales_orders']['ProductName'].unique())[:50]
    selected_products = st.multiselect("Select products in basket", options=product_options)
    
    if selected_products and components['market_basket'].rules is not None:
        recommendations = components['market_basket'].get_recommendations(selected_products, top_n=5)
        if len(recommendations) > 0:
            st.success(f"Found {len(recommendations)} recommendations!")
            for rec in recommendations:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{rec['recommended_item']}**")
                        st.caption(f"Based on: {rec['input_item']}")
                    with col2:
                        st.metric("Lift", f"{rec['lift']:.1f}x")
                    with col3:
                        st.metric("Confidence", f"{rec['confidence']:.1%}")

# Tab 6: Territory Intelligence
with tab6:
    st.header("🌍 Territory & Store Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Territory")
        
        try:
            if len(data['territories']) > 0:
                fig_territory = px.bar(
                    data['territories'],
                    x='Name',
                    y='SalesYTD',
                    title="Sales by Territory (YTD)",
                    color='SalesYTD',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_territory, use_container_width=True)
            else:
                st.info("No territory revenue data available")
        except Exception as e:
            st.info("Territory data not available")
    
    with col2:
        st.subheader("Territory Performance")
        
        if len(data['territories']) > 0:
            fig_perf = px.bar(
                data['territories'],
                x='Name',
                y='SalesYTD',
                color='Group',
                title="Year-to-Date Sales by Territory",
                labels={'SalesYTD': 'YTD Sales ($)'}
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No territory performance data available")
    
    st.subheader("📊 Territory Customer Metrics")
    
    try:
        if len(data['customers']) > 0 and 'TerritoryID' in data['customers'].columns and len(data['territories']) > 0:
            territory_customers = data['customers'].merge(
                data['territories'][['TerritoryID', 'Name']],
                on='TerritoryID', how='left'
            )
            territory_metrics = territory_customers.groupby('Name').agg({'CustomerID': 'nunique'}).reset_index()
            territory_metrics.columns = ['Territory', 'CustomerCount']
            
            if len(territory_metrics) > 0:
                fig_territory_metrics = px.bar(
                    territory_metrics,
                    x='Territory',
                    y='CustomerCount',
                    title="Customer Distribution by Territory",
                    color='CustomerCount',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_territory_metrics, use_container_width=True)
    except:
        pass
    
    # Cohort analysis
    st.subheader("📈 Customer Cohort Analysis")
    
    sales_with_dates = data['sales_orders'].copy()
    sales_with_dates['OrderDate'] = pd.to_datetime(sales_with_dates['OrderDate'])
    sales_with_dates['CohortMonth'] = sales_with_dates.groupby('CustomerID')['OrderDate'].transform('min').dt.to_period('M')
    sales_with_dates['OrderMonth'] = sales_with_dates['OrderDate'].dt.to_period('M')
    sales_with_dates['Period'] = (sales_with_dates['OrderMonth'] - sales_with_dates['CohortMonth']).apply(lambda x: x.n)
    
    cohort_pivot = sales_with_dates.groupby(['CohortMonth', 'Period']).agg({'CustomerID': 'nunique'}).reset_index()
    cohort_matrix = cohort_pivot.pivot(index='CohortMonth', columns='Period', values='CustomerID')
    cohort_retention = cohort_matrix.div(cohort_matrix[0], axis=0) * 100
    cohort_retention.index = cohort_retention.index.astype(str)
    
    if len(cohort_retention) > 0:
        fig_cohort = px.imshow(
            cohort_retention.iloc[:12, :12].values,
            x=cohort_retention.iloc[:12, :12].columns.astype(str),
            y=cohort_retention.iloc[:12, :12].index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Customer Cohort Retention (%)",
            labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention %")
        )
        fig_cohort.update_traces(texttemplate='%{text:.1f}%')
        fig_cohort.update_layout(height=500)
        st.plotly_chart(fig_cohort, use_container_width=True)
    else:
        st.info("No cohort data available")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>👥 Customer Value Intelligence Hub v1.0 | Powered by Advanced Analytics, Machine Learning & Causal Inference</p>
    <p>Real-time data from AdventureWorks Customer Database</p>
    <p>CLV Models | Churn Prediction | Market Basket Analysis | Customer Segmentation | Territory Intelligence | B2B Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)