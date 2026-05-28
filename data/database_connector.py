import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from typing import Optional, Dict, List
import logging
from config import settings
import urllib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerDatabaseConnector:
    def __init__(self):
        self.connection_string = settings.DATABASE_URL
        self.engine = self._create_engine()
        
    def _create_engine(self):
        try:
            params = urllib.parse.quote_plus(self.connection_string)
            engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={params}",
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            logger.info("Database engine created successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_sales_orders(self) -> pd.DataFrame:
        query = """
        SELECT 
            soh.SalesOrderID,
            soh.OrderDate,
            soh.DueDate,
            soh.ShipDate,
            soh.Status,
            soh.OnlineOrderFlag,
            soh.CustomerID,
            soh.SalesPersonID,
            soh.TerritoryID,
            soh.SubTotal,
            soh.TaxAmt,
            soh.Freight,
            soh.TotalDue,
            sod.SalesOrderDetailID,
            sod.ProductID,
            sod.OrderQty,
            sod.UnitPrice,
            sod.UnitPriceDiscount,
            sod.LineTotal,
            p.Name AS ProductName,
            p.ProductNumber,
            ps.Name AS SubcategoryName,
            pc.Name AS CategoryName
        FROM Sales.SalesOrderHeader soh
        INNER JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
        INNER JOIN Production.Product p ON sod.ProductID = p.ProductID
        LEFT JOIN Production.ProductSubcategory ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID
        LEFT JOIN Production.ProductCategory pc ON ps.ProductCategoryID = pc.ProductCategoryID
        ORDER BY soh.OrderDate DESC
        """
        return self.execute_query(query)
    
    def get_customers(self) -> pd.DataFrame:
        query = """
        SELECT 
            c.CustomerID,
            c.PersonID,
            c.StoreID,
            c.TerritoryID,
            c.AccountNumber,
            p.FirstName,
            p.LastName,
            p.EmailPromotion,
            ea.EmailAddress,
            a.City,
            a.PostalCode,
            sp.Name AS StateProvince,
            cr.Name AS CountryRegion,
            st.Name AS TerritoryName,
            st.[Group] AS TerritoryGroup
        FROM Sales.Customer c
        LEFT JOIN Person.Person p ON c.PersonID = p.BusinessEntityID
        LEFT JOIN Person.EmailAddress ea ON p.BusinessEntityID = ea.BusinessEntityID
        LEFT JOIN Person.BusinessEntityAddress bea ON p.BusinessEntityID = bea.BusinessEntityID
        LEFT JOIN Person.Address a ON bea.AddressID = a.AddressID
        LEFT JOIN Person.StateProvince sp ON a.StateProvinceID = sp.StateProvinceID
        LEFT JOIN Person.CountryRegion cr ON sp.CountryRegionCode = cr.CountryRegionCode
        LEFT JOIN Sales.SalesTerritory st ON c.TerritoryID = st.TerritoryID
        """
        return self.execute_query(query)
    
    def get_customer_demographics(self) -> pd.DataFrame:
        query = """
        SELECT 
            BusinessEntityID,
            TotalPurchaseYTD,
            DateFirstPurchase,
            BirthDate,
            MaritalStatus,
            YearlyIncome,
            Gender,
            TotalChildren,
            NumberChildrenAtHome,
            Education,
            Occupation,
            HomeOwnerFlag,
            NumberCarsOwned
        FROM Sales.vPersonDemographics
        """
        return self.execute_query(query)
    
    def get_sales_territories(self) -> pd.DataFrame:
        query = """
        SELECT 
            TerritoryID,
            Name,
            CountryRegionCode,
            [Group],
            SalesYTD,
            SalesLastYear,
            CostYTD,
            CostLastYear
        FROM Sales.SalesTerritory
        """
        return self.execute_query(query)
    
    def get_special_offers(self) -> pd.DataFrame:
        query = """
        SELECT 
            so.SpecialOfferID,
            so.Description,
            so.DiscountPct,
            so.Type,
            so.Category,
            so.StartDate,
            so.EndDate,
            so.MinQty,
            so.MaxQty,
            sop.ProductID
        FROM Sales.SpecialOffer so
        LEFT JOIN Sales.SpecialOfferProduct sop ON so.SpecialOfferID = sop.SpecialOfferID
        """
        return self.execute_query(query)
    
    def get_sales_reasons(self) -> pd.DataFrame:
        query = """
        SELECT 
            sohsr.SalesOrderID,
            sr.Name AS SalesReason,
            sr.ReasonType
        FROM Sales.SalesOrderHeaderSalesReason sohsr
        INNER JOIN Sales.SalesReason sr ON sohsr.SalesReasonID = sr.SalesReasonID
        """
        return self.execute_query(query)
    
    def get_sales_persons(self) -> pd.DataFrame:
        query = """
        SELECT 
            sp.BusinessEntityID,
            sp.TerritoryID,
            sp.SalesQuota,
            sp.Bonus,
            sp.CommissionPct,
            sp.SalesYTD,
            sp.SalesLastYear,
            p.FirstName,
            p.LastName,
            e.JobTitle
        FROM Sales.SalesPerson sp
        INNER JOIN Person.Person p ON sp.BusinessEntityID = p.BusinessEntityID
        INNER JOIN HumanResources.Employee e ON sp.BusinessEntityID = e.BusinessEntityID
        """
        return self.execute_query(query)