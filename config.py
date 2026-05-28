from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "DRIVER={ODBC Driver 17 for SQL Server};SERVER=YOUR_SERVER_NAME;DATABASE=YOUR_DATABASE_NAME;Trusted_Connection=yes;")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Customer analytics settings - B2B focused
    CHURN_WINDOW_DAYS: int = 365  
    HIGH_CHURN_WINDOW_DAYS: int = 730  
    HIGH_VALUE_THRESHOLD: float = 5000.0
    RFM_RECENCY_WEIGHT: float = 0.3
    RFM_FREQUENCY_WEIGHT: float = 0.3
    RFM_MONETARY_WEIGHT: float = 0.4
    SEGMENT_MIN_SIZE: int = 30
    MARKET_BASKET_MIN_SUPPORT: float = 0.01
    MARKET_BASKET_MIN_LIFT: float = 1.5
    
    # B2B specific settings
    TOP_CUSTOMER_PERCENTILE: float = 0.05  
    REPEAT_ORDER_THRESHOLD: int = 2 
    CONCENTRATION_THRESHOLD: float = 80.0  
    
    # Cache settings
    CACHE_TTL_SECONDS: int = 3600
    MAX_CACHE_ENTRIES: int = 200

settings = Settings()