# Azure Demand Forecasting & Capacity Optimization System

## Project Overview

This project forecasts Azure resource demand (Compute and Storage) and supports capacity planning using time-series and machine learning methods.

It includes:

- demand forecasting models
- batch prediction pipeline
- interactive Streamlit dashboard for insights

## Live Dashboard

https://azure-demand.streamlit.app/

## Objectives

- Forecast Azure usage across regions and service types
- Detect demand patterns and seasonality
- Support capacity optimization and cost control
- Turn model outputs into operational insights via dashboard

## Dataset

The dataset is multivariate and includes:

- timestamp
- region
- service_type
- usage_units
- provisioned_capacity
- cost_usd
- availability_pct
- is_holiday
- economic_index
- market_demand_index
- temperature_index

## Methodology

1. Data Preparation:

- Parse and sort timestamps
- Standardize categories (region/service names)
- Handle missing values and duplicates

2. Feature Engineering:

- Time-based features (hour, day, weekday, month, year)
- Lag features and rolling statistics
- Spike indicators and encoded categorical features

3. Modeling:

- Baseline ARIMA
- XGBoost regressor (tuned model used for deployment)
- Evaluation with RMSE and MAE

4. Deployment and Usage:

- Batch forecasts from CSV input
- Visual analytics in Streamlit dashboard
- Streamlit dashboard used as the Milestone 4 deliverable

## Milestone Breakdown

### Milestone 1: Data Collection and Cleaning

- Loaded raw Azure demand data and validated core schema
- Converted timestamp fields and ensured chronological ordering
- Standardized categorical values (region and service naming)
- Removed duplicates and handled missing values
- Produced a clean, analysis-ready dataset

### Milestone 2: Feature Engineering and Exploratory Insights

- Created time-based features (hour, day, weekday, month, year)
- Added lag features and rolling aggregates for trend capture
- Created spike/risk indicators for unusual usage behavior
- Encoded categorical variables for model compatibility
- Checked data quality after transformation and validated feature readiness

### Milestone 3: Model Development and Evaluation

- Built baseline ARIMA model for time-series comparison
- Trained XGBoost regression model with engineered features
- Performed hyperparameter tuning for improved performance
- Compared models using RMSE and MAE
- Selected tuned XGBoost as primary production model and stored artifacts

### Milestone 4: Forecast Integration and Operationalization

- Built Streamlit dashboard for KPI tracking, regional trends, and risk alerts
- Integrated forecast outputs into dashboard visuals for decision support
- Added filters and KPI cards for region/service-level analysis
- Published the dashboard for live access
