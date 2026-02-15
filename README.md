# Azure Demand Forecasting & Capacity Optimization System

## 📌 Overview

This project builds a predictive system to forecast Azure Compute and Storage demand using time-series analysis and machine learning. The goal is to help cloud infrastructure teams make smarter capacity provisioning decisions, reduce over-allocation costs, and improve service reliability.

---

## 🎯 Objectives

- Forecast future Azure resource demand across regions
- Identify seasonal and regional usage patterns
- Optimize capacity provisioning and reduce waste
- Incorporate external factors affecting cloud demand

---

## 📂 Dataset

A synthetic multivariate dataset was generated to simulate Azure cloud usage.
It includes:

**Core Features**

- Timestamp (daily)
- Region
- Service type (Compute / Storage)
- Usage units
- Provisioned capacity
- Cost
- Availability percentage

**External Variables**

- Holiday indicator
- Economic activity index
- Market demand index
- Seasonal temperature proxy

These variables allow both univariate and multivariate forecasting experiments.

---

## ⚙️ Methodology

### 1️⃣ Data Preparation

- Timestamp formatting and sorting
- Region standardization
- Duplicate removal
- Missing value handling using interpolation and forward fill

### 2️⃣ Feature Engineering

- Time features (month, weekday)
- Utilization ratio
- Lag features for demand history
- Rolling averages for trend smoothing

### 3️⃣ Modeling

Models evaluated:

- ARIMA (baseline time-series model)
- Tree-based models for multivariate forecasting
- Performance measured using MAE and RMSE

### 4️⃣ Output & Insights

- Demand forecasts by region and service
- Capacity buffer recommendations
- Demand vs provision comparison
- Visual trend and seasonality analysis

---

## 🧠 Key Learnings

- Cloud demand behaves like other infrastructure loads with strong seasonal signals
- External indicators improve forecasting accuracy
- Data quality and preprocessing strongly impact model performance

