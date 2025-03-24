# Sales Forecasting

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Regression - Intermediate</li>
      <!-- <li><strong>Estimated Time</strong>: 4-6 hours</li> -->
      <li><strong>Skills</strong>: Time series analysis, feature engineering, regression models</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/en/core/regression/linear-regression.html">Regression Analysis</a></li>
    </ul>
  </div>
</div>

## Project Background

Sales forecasting is a key task in business operations. Accurate forecasting can help businesses optimize inventory management, human resources planning, and marketing strategies. By analyzing historical sales data and related factors (such as promotional activities, seasonality, price changes, etc.), we can build models to predict future sales performance.

In this project, we will use retail sales data to build a sales forecasting model, helping businesses make more informed decisions.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did you know?
  </div>
  <div class="knowledge-card__content">
    <p>Large retailers like Walmart handle millions of transactions every day. They use advanced forecasting models to optimize inventory and pricing. Accurate sales forecasting is estimated to save large retailers millions of dollars in inventory costs and significantly reduce product waste annually.</p>
  </div>
</div>

## Dataset Introduction

We will use a retail sales dataset containing daily sales records for a supermarket chain over two years. The dataset includes the following features:

- Date: Sales date
- Store: Store ID
- Item: Item ID
- Sales: Quantity sold
- Price: Item price
- Promotion: Whether there was a promotion (1 for yes, 0 for no)
- Holiday: Whether it was a holiday (1 for yes, 0 for no)
- Temperature: Temperature on the day
- Fuel_Price: Fuel price
- CPI: Consumer Price Index
- Unemployment: Unemployment rate

The target variable is Sales (quantity sold). Our task is to forecast the sales for a future period.

## Project Goals

1. Build a regression model that can accurately predict sales
2. Identify the key factors affecting sales
3. Analyze the impact of seasonality and trends on sales
4. Evaluate the performance of different forecasting methods
5. Provide visualizations and business recommendations based on sales forecasts

## Implementation Steps

### 1. Data Exploration and Visualization

First, we need to understand the basic features and distributions of the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
df = pd.read_csv('retail_sales.csv')

# Convert the Date column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# View basic data information
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the sales trend
plt.figure(figsize=(12, 6))
sales_by_date = df.groupby('Date')['Sales'].sum().reset_index()
plt.plot(sales_by_date['Date'], sales_by_date['Sales'])
plt.title('Total Sales Trend by Date')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# View the seasonality pattern in sales
plt.figure(figsize=(12, 6))
sales_by_month = df.groupby(df['Date'].dt.month)['Sales'].mean().reset_index()
plt.bar(sales_by_month['Date'], sales_by_month['Sales'])
plt.title('Average Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(1, 13))
plt.grid(True, axis='y')
plt.show()

# View the impact of promotions on sales
plt.figure(figsize=(10, 6))
sns.boxplot(x='Promotion', y='Sales', data=df)
plt.title('Impact of Promotion on Sales')
plt.xlabel('Promotion')
plt.ylabel('Sales')
plt.show()

# View the impact of temperature on sales
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['Sales'], alpha=0.5)
plt.title('Temperature vs Sales')
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```

### 2. Feature Engineering

Create new features that will assist with forecasting:

```python
# Extract time features from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Create season features
df['Season'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                          2 if x in [3, 4, 5] else 
                                          3 if x in [6, 7, 8] else 4)

# Create weekend flag feature
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create lag features (previous day, previous week sales)
df_lag = df.copy()
df_lag = df_lag.sort_values(['Store', 'Item', 'Date'])
df_lag['Sales_Lag1'] = df_lag.groupby(['Store', 'Item'])['Sales'].shift(1)
df_lag['Sales_Lag7'] = df_lag.groupby(['Store', 'Item'])['Sales'].shift(7)

# Remove rows with NaN values (due to lag feature creation)
df_lag = df_lag.dropna()

# View new features
print(df_lag.head())
```

### 3. Time Series Decomposition

Analyze the trend, seasonality, and residuals of the sales data:

```python
# Select sales data for a specific store and item for time series decomposition
store_item_sales = df[(df['Store'] == 1) & (df['Item'] == 1)].set_index('Date')['Sales']

# Ensure the index is evenly spaced
store_item_sales = store_item_sales.asfreq('D')

# Time series decomposition
decomposition = seasonal_decompose(store_item_sales, model='multiplicative', period=7)

# Visualize decomposition results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonality')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

### 4. Model Training and Evaluation

We will try multiple regression models and compare their performance:

```python
# Prepare features and target variable
X = df_lag.drop(['Date', 'Sales'], axis=1)
y = df_lag['Sales']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['Store', 'Item', 'Season'], drop_first=True)

# Split into training and testing sets (time-based split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"{model_name}:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.2f}")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return model, y_test_pred

# Linear Regression
lr_model, lr_pred = evaluate_model(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression Model")
print("\n")

# Random Forest Regression
rf_model, rf_pred = evaluate_model(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), 
                                  X_train, X_test, y_train, y_test, "Random Forest Regression Model")
```

### 5. Feature Importance Analysis

Understand which factors have the most impact on sales:

```python
# Analyze feature importance using the Random Forest model
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Display the top 15 most important features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance (Top 15)')
plt.tight_layout()
plt.show()
```

### 6. Prediction Visualization

Visualize the comparison between model predictions and actual values:

```python
# Visualize prediction results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Sales')
plt.plot(rf_pred, label='Predicted Sales', alpha=0.7)
plt.title('Random Forest Model: Actual Sales vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Prediction error analysis
errors = y_test - rf_pred
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50)
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

## Advanced Challenges

If you have completed the basic tasks, you can try the following advanced challenges:

1. **Advanced Time Series Models**: Try using ARIMA, SARIMA, or Prophet models specifically designed for time series
2. **Multi-step Prediction**: Build models that forecast sales for multiple future time points
3. **Ensemble Forecasting**: Combine predictions from multiple models to improve forecasting accuracy
4. **Feature Selection**: Use feature selection techniques to find the most predictive subset of features
5. **Cross-Validation**: Implement time series cross-validation to get more robust model evaluations

## Conclusion and Reflection

Through this project, we learned how to build a sales forecasting model, from data exploration to model evaluation. We found that promotional activities, seasonality, and historical sales data significantly impact the prediction of future sales.

In practice, sales forecasting models help businesses optimize inventory management, reduce inventory costs, and minimize product waste while ensuring product availability to meet customer demand. Additionally, accurate sales forecasts can help businesses develop more effective marketing strategies and promotion plans.

### Reflective Questions

1. Besides the features we used, what other factors might affect sales? How can we obtain this data?
2. In what situations might sales forecasting models fail? How can we address these challenges?
3. How can sales forecasting results be translated into specific business decisions and actions?

<div class="practice-link">
  <a href="/en/projects/regression/anomaly-detection.html" class="button">Next Project: Anomaly Detection and Prediction</a>
</div>
