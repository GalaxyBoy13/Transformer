# Transformer

This is a nascent project for Cognizant organised Digital Nurture Technoverse 2024

## Project Overview
This project focuses on demand forecasting and inventory management using advanced machine learning techniques. We leverage Long Short-Term Memory (LSTM) networks for time-series forecasting and the FP-Growth algorithm for market basket analysis. The primary dataset used is the DataCo Supply Chain dataset.

### Key Features

Demand Forecasting with LSTM:

* Preprocessing: Data cleaning, handling missing values, and feature engineering.
* Model Building: Construction and training of LSTM models for each product to forecast future demand.
* Forecasting: Predicting demand for the next 7 days and calculating safety stock and reorder points for optimal inventory management.

Market Basket Analysis with FP-Growth:

* Data Preparation: Grouping and one-hot encoding of transaction data.
* Frequent Pattern Mining: Using the FP-Growth algorithm to identify frequent itemsets.
* Association Rules: Generating and ranking association rules based on support, confidence, and lift to uncover product pairings and drive cross-selling strategies.

Data Visualization:

* Interactive visualizations using Plotly to analyze and present sales data by category, region, and other dimensions.

### Steps

* Data Cleaning and Preprocessing
  * Removed unwanted columns to focus on relevant features.
  * Handled missing values and duplicates to ensure data quality.
  * Converted date columns to appropriate datetime formats and handled outliers.
* Modeling and Forecasting
  * Utilized LSTM networks to forecast daily units sold for individual products.
  * Calculated demand rate, safety stock, and reorder points to aid in inventory management.
  * Grouped products by category to analyze and compare demand rates across categories.
* Market Basket Analysis
  * Implemented the FP-Growth algorithm for efficient frequent pattern mining.
  * Generated association rules to identify potential product pairings.
  * Calculated a composite score for each rule to rank based on confidence and lift.
* Visualizations
  * Created bar charts to showcase aggregate sales by category and region.
  * Analyzed sales performance specifically for the Indian market.
* Results and Insights
  * Detailed insights into product demand forecasting and inventory management.
  * Identification of key product pairings and cross-selling opportunities.
  * Comprehensive visual analysis of sales data to support decision-making.

### Conclusion

This project demonstrates the application of LSTM for demand forecasting and FP-Growth for market basket analysis, providing actionable insights for effective inventory management and sales strategy optimization.
