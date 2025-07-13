-- Eliminazione tabelle in ordine corretto
DROP TABLE IF EXISTS SALE;
DROP TABLE IF EXISTS "ORDER";
DROP TABLE IF EXISTS PRODUCT;
DROP TABLE IF EXISTS CUSTOMER;
DROP TABLE IF EXISTS ECONOMIC_MEASUREMENT;
DROP TABLE IF EXISTS COUNTRY;

-- Creazione delle tabelle

CREATE TABLE COUNTRY (
    CountryCode CHAR(3) PRIMARY KEY,
    CountryName VARCHAR(100) NOT NULL
);

CREATE TABLE ECONOMIC_MEASUREMENT (
    CountryCode CHAR(3),
    Year INT,
    GDP DECIMAL(15,2),
    GDP_per_capita DECIMAL(10,2),
    PRIMARY KEY (CountryCode, Year),
    FOREIGN KEY (CountryCode) REFERENCES COUNTRY(CountryCode)
);

CREATE TABLE CUSTOMER (
    CustomerID INT PRIMARY KEY,
    CountryCode CHAR(3),
    FOREIGN KEY (CountryCode) REFERENCES COUNTRY(CountryCode)
);

CREATE TABLE PRODUCT (
    StockCode VARCHAR(50) PRIMARY KEY,
    ArticleName VARCHAR(255),
    UnitPrice DECIMAL(10,2),
    Category VARCHAR(100),
    EstimatedUnitCost DECIMAL(10,2)
);

CREATE TABLE "ORDER" (
    InvoiceNo INT PRIMARY KEY,
    CustomerID INT,
    InvoiceDate DATE,
    PaymentMethod VARCHAR(50),
    ShippingCost DECIMAL(10,2),
    SalesChannel VARCHAR(50),
    ShipmentProvider VARCHAR(100),
    WarehouseLocation VARCHAR(100),
    OrderPriority VARCHAR(20),
    FOREIGN KEY (CustomerID) REFERENCES CUSTOMER(CustomerID)
);

CREATE TABLE SALE (
    InvoiceNo INT,
    StockCode VARCHAR(50),
    Quantity INT,
    Discount DECIMAL(5,2),
    ReturnStatus VARCHAR(20),
    PRIMARY KEY (InvoiceNo, StockCode),
    FOREIGN KEY (InvoiceNo) REFERENCES "ORDER"(InvoiceNo),
    FOREIGN KEY (StockCode) REFERENCES PRODUCT(StockCode)
);

-- Indici

CREATE INDEX idx_customer_country ON CUSTOMER(CountryCode);
CREATE INDEX idx_economic_measurement_country ON ECONOMIC_MEASUREMENT(CountryCode);
CREATE INDEX idx_order_customer ON "ORDER"(CustomerID);
CREATE INDEX idx_order_date ON "ORDER"(InvoiceDate);
CREATE INDEX idx_sale_invoice ON SALE(InvoiceNo);
CREATE INDEX idx_sale_stock ON SALE(StockCode);

ALTER TABLE ECONOMIC_MEASUREMENT 
ALTER COLUMN GDP TYPE DECIMAL(20,2);

ALTER TABLE ECONOMIC_MEASUREMENT 
ALTER COLUMN GDP_per_capita TYPE DECIMAL(15,2);

