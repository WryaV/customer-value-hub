markdown

# Customer Value Intelligence Hub

A comprehensive customer analytics dashboard that transforms raw transaction data into actionable business intelligence, combining Streamlit web development with advanced data analytics (CLV modeling, churn prediction, market basket analysis, and customer segmentation).

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Customer Genome Map | UMAP clustering visualizing 19,000+ customers by behavior patterns |
| Lifetime Value (CLV) | Pareto/NBD model predicting customer lifetime value with confidence intervals |
| Churn War Room | Risk factor identification + retention ROI simulator |
| Promotion Intelligence | Discount effectiveness analysis + persuasion quadrant segmentation |
| Product Ecosystem | Market basket analysis + product affinity chord diagram |
| Territory Intelligence | Regional performance + cohort retention heatmaps |

## Architecture

CustomerValue/
├── config.py # Settings & thresholds
├── main.py # Streamlit entry point
├── requirements.txt # Dependencies
├── data/
│ ├── database_connector.py # SQL Server connection
│ └── feature_engineering.py # RFM, entropy, trend analysis
├── models/customer/
│ ├── clv_model.py # Pareto/NBD, health scores
│ ├── churn_model.py # Survival analysis, risk prediction
│ ├── segmentation_model.py # UMAP, HDBSCAN, K-means
│ └── market_basket.py # Apriori, association rules
├── visualizations/
│ ├── plotly_components.py # RFM cube, chord diagrams
│ └── custom_charts.py # Uplift curves, cohort heatmaps
├── utils/
│ ├── statistical_tests.py # Chi-square, t-test, ANOVA
│ └── metrics.py # CAC, NRR, cohort metrics
└── dashboards/
└── customer_app.py # Main Streamlit dashboard
text


## Key Metrics Calculated

- Customer Lifetime Value (CLV) with confidence intervals
- Churn probability & risk segmentation
- RFM (Recency, Frequency, Monetary) scores
- Customer health scores (0-100)
- Revenue concentration (Pareto analysis)
- Product affinity (lift & confidence)
- Cohort retention rates
- Promotion ROI simulation

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python 3.11 |
| Database | SQL Server (AdventureWorks2014) |
| ML/Analytics | scikit-learn, UMAP, HDBSCAN, lifelines, mlxtend |
| Visualization | Plotly, Plotly Express |
| Data Processing | Pandas, NumPy, SciPy |

## Installation

### Prerequisites

- Python 3.11 or higher
- SQL Server with AdventureWorks2014 database
- ODBC Driver 17 for SQL Server

### Setup

1. Clone the repository
```bash
git clone 
cd CustomerValue

    Create virtual environment

bash

python -m venv .venv
.venv\Scripts\activate

    Install dependencies

bash

pip install -r requirements.txt

    Configure database connection in .env file

text

DATABASE_URL=DRIVER={ODBC Driver 17 for SQL Server};SERVER=YOUR_SERVER;DATABASE=AdventureWorks2014;Trusted_Connection=yes;

    Run the dashboard

bash

streamlit run dashboards/customer_app.py --server.port 8502