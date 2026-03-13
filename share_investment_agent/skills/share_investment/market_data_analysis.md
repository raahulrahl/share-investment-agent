# Market Data Analysis Specialist

## Overview
The Market Data Analysis Specialist is responsible for collecting and preprocessing all necessary market data for A-share investment analysis. This specialist serves as the foundation for all subsequent analysis by providing comprehensive and accurate market information.

## Core Responsibilities

### Data Collection
- **Real-time Market Data**: Current stock prices, trading volumes, and market indicators
- **Historical Price Data**: Daily OHLCV data for technical analysis
- **Financial Metrics**: Company financial indicators and ratios
- **Market Information**: Market capitalization, trading statistics, and sector data

### Data Sources
- **Akshare API**: Primary source for Chinese A-share market data
- **Real-time Quotes**: Live market data from major exchanges
- **Financial Statements**: Quarterly and annual financial reports
- **Market Indices**: Relevant market benchmarks and sector indices

### Data Processing
- **Data Validation**: Ensure data completeness and accuracy
- **Standardization**: Convert all data to consistent formats
- **Quality Control**: Handle missing values and data anomalies
- **Timestamp Management**: Proper date/time handling for analysis

## Key Data Types

### Price Data
```python
{
    "date": "YYYY-MM-DD",
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": int,
    "amount": float
}
```

### Financial Metrics
```python
{
    "market_cap": float,
    "pe_ratio": float,
    "pb_ratio": float,
    "revenue": float,
    "net_income": float,
    "roe": float,
    "debt_ratio": float
}
```

### Market Information
```python
{
    "ticker": str,
    "company_name": str,
    "sector": str,
    "industry": str,
    "listing_date": str,
    "total_shares": int,
    "float_shares": int
}
```

## Analysis Workflow

### 1. Data Acquisition
- Connect to Akshare API endpoints
- Retrieve real-time and historical data
- Fetch financial statements and metrics
- Collect market-related information

### 2. Data Processing
- Clean and validate collected data
- Handle missing or corrupted data points
- Standardize data formats across sources
- Create unified data package

### 3. Quality Assurance
- Verify data completeness
- Check for data consistency
- Validate against known benchmarks
- Document any data limitations

## Integration Points

### Downstream Specialists
- **Technical Analysis Specialist**: Provides price history and volume data
- **Fundamental Analysis Specialist**: Supplies financial metrics and statements
- **Sentiment Analysis Specialist**: Offers market context and news data
- **Valuation Specialist**: Delivers company financial information

### Data Formats
- **JSON Format**: Structured data for API consumption
- **DataFrame Format**: Tabular data for numerical analysis
- **Time Series**: Chronological data for trend analysis

## Quality Metrics

### Data Freshness
- Real-time data: < 1 minute delay
- Historical data: Complete daily records
- Financial data: Latest quarterly reports

### Accuracy Standards
- Price accuracy: 4 decimal places
- Volume accuracy: Integer values
- Financial metrics: 2 decimal places

## Error Handling

### Common Issues
- API rate limiting
- Data source unavailability
- Missing data points
- Format inconsistencies

### Resolution Strategies
- Implement retry mechanisms
- Use backup data sources
- Apply data interpolation
- Maintain error logs

## Performance Considerations

### Optimization Techniques
- Data caching strategies
- Parallel API calls
- Efficient data structures
- Memory management

### Scalability
- Handle multiple stock requests
- Process large datasets
- Manage concurrent connections
- Optimize query performance

## Compliance and Regulations

### Data Usage
- Respect data provider terms
- Maintain data privacy
- Follow market regulations
- Ensure proper attribution

### Best Practices
- Regular data validation
- Transparent data sourcing
- Comprehensive documentation
- Audit trail maintenance
