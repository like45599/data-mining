# E-commerce User Data Cleaning and Analysis

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Entry-level</li>
      <li><strong>Type</strong>: Data cleaning and preprocessing</li>
      <!-- <li><strong>Estimated Time</strong>: 3-5 hours</li> -->
      <li><strong>Skills</strong>: Missing value treatment, anomaly detection, data transformation, feature engineering</li>
      <li><strong>Corresponding Knowledge Module</strong>: <a href="/core/preprocessing/data-presentation.html">Data Preprocessing</a></li>
    </ul>
  </div>
</div>

## Project Background

E-commerce platforms generate massive amounts of user behavior data daily, including activities such as browsing, searching, adding to cart, and purchasing. This data is critical for understanding user behavior patterns, optimizing product recommendations, and improving conversion rates. However, raw data often contains missing values, outliers, and inconsistent formats, which need to be cleaned and preprocessed before further analysis.

In this project, we will process a dataset of user behavior from an e-commerce platform, performing data cleaning and preprocessing to prepare for subsequent user behavior analysis.

<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Data scientists typically spend 60-70% of their time on data cleaning and preprocessing. High-quality data preprocessing not only improves model performance but also reduces errors and biases in subsequent analyses.</p>
  </div>
</div>

## Dataset Introduction

The dataset used in this project contains one week of user behavior data from an e-commerce platform, consisting of 10,000 records with the following fields:

- **user_id**: User ID
- **session_id**: Session ID
- **timestamp**: Activity timestamp
- **page_url**: URL of the visited page
- **event_type**: Event type (view, cart, purchase)
- **product_id**: Product ID
- **category**: Product category
- **price**: Product price
- **user_agent**: User browser information
- **user_region**: User region

The dataset contains various data quality issues, including missing values, outliers, inconsistent formats, and duplicate records.

## Project Objectives

1. Identify and handle missing values in the dataset.
2. Detect and handle outliers and anomalies.
3. Standardize and transform data formats.
4. Create meaningful derived features.
5. Prepare a clean dataset for subsequent analysis.

## Implementation Steps

### Step 1: Data Loading and Preliminary Exploration

First, we load the data and perform an initial exploration to understand its basic characteristics.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('ecommerce_data.csv')

# View basic information about the data
print(df.info())
print(df.describe())

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check for duplicate records
print(f"Number of duplicate records: {df.duplicated().sum()}")
```

### Step 2: Handling Missing Values

Based on our preliminary exploration, we found missing values in the dataset that need to be handled with appropriate strategies.

```python
# Check the percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)

# Handle missing product prices – fill with the median price of products in the same category
df['price'] = df.groupby('category')['price'].transform(lambda x: x.fillna(x.median()))

# Handle missing user regions – fill with the mode
most_common_region = df['user_region'].mode()[0]
df['user_region'] = df['user_region'].fillna(most_common_region)

# Handle missing product categories – infer from product_id
# Assume we have a product mapping dictionary
product_category_map = {...}  # In a real project, this mapping needs to be constructed
df['category'] = df.apply(lambda row: product_category_map.get(row['product_id'], row['category']) 
                         if pd.isnull(row['category']) else row['category'], axis=1)

# Drop records missing critical information
df = df.dropna(subset=['user_id', 'session_id', 'timestamp', 'event_type'])

# Check the missing values after processing
print(df.isnull().sum())
```

### Step 3: Detecting and Handling Outliers

Next, we detect and handle outliers in the dataset.

```python
# Check the distribution of prices
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['price'])
plt.title('Price Distribution')
plt.show()

# Detect price outliers using the IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
price_outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound))
print(f"Number of price outliers: {price_outliers.sum()}")

# Handle price outliers – clip extreme values to a reasonable range
df['price_cleaned'] = df['price'].clip(lower_bound, upper_bound)

# Validate timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
invalid_timestamps = df['timestamp'].isnull()
print(f"Number of invalid timestamps: {invalid_timestamps.sum()}")

# Drop records with invalid timestamps
df = df.dropna(subset=['timestamp'])

# Check the validity of event types
valid_event_types = ['view', 'cart', 'purchase']
invalid_events = ~df['event_type'].isin(valid_event_types)
print(f"Number of invalid event types: {invalid_events.sum()}")

# Correct event types – assume some rules for correction
df.loc[df['event_type'] == 'add_to_cart', 'event_type'] = 'cart'
df.loc[df['event_type'] == 'buy', 'event_type'] = 'purchase'
```

### Step 4: Data Standardization and Format Conversion

To facilitate subsequent analysis, we standardize data formats.

```python
# Standardize timestamp format
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Extract page type from URL
def extract_page_type(url):
    if 'product' in url:
        return 'product_page'
    elif 'category' in url:
        return 'category_page'
    elif 'search' in url:
        return 'search_page'
    elif 'cart' in url:
        return 'cart_page'
    elif 'checkout' in url:
        return 'checkout_page'
    else:
        return 'other'

df['page_type'] = df['page_url'].apply(extract_page_type)

# Standardize product category
df['category'] = df['category'].str.lower().str.strip()

# Extract device type from user_agent
def extract_device_type(user_agent):
    if pd.isnull(user_agent):
        return 'unknown'
    user_agent = user_agent.lower()
    if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

df['device_type'] = df['user_agent'].apply(extract_device_type)
```

### Step 5: Creating Derived Features

To enhance the analytical value of the data, we create several derived features.

```python
# Calculate session duration
session_start = df.groupby('session_id')['timestamp'].min()
session_end = df.groupby('session_id')['timestamp'].max()
session_duration = (session_end - session_start).dt.total_seconds() / 60  # Convert to minutes

# Merge session duration into the original dataframe
session_duration_df = pd.DataFrame({'session_duration': session_duration})
session_duration_df.reset_index(inplace=True)
df = pd.merge(df, session_duration_df, on='session_id', how='left')

# Count the number of activities per session for each event type
activity_count = df.groupby(['session_id', 'event_type']).size().unstack(fill_value=0)
activity_count.columns = [f'{col}_count' for col in activity_count.columns]
activity_count.reset_index(inplace=True)
df = pd.merge(df, activity_count, on='session_id', how='left')

# Create a purchase flag
df['has_purchase'] = df['session_id'].isin(df[df['event_type'] == 'purchase']['session_id'])

# Calculate price ranges
price_bins = [0, 10, 50, 100, 500, float('inf')]
price_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
df['price_range'] = pd.cut(df['price_cleaned'], bins=price_bins, labels=price_labels)
```

### Step 6: Data Validation and Export

Finally, we validate the cleaned data and export it for further analysis.

```python
# Check data quality
print("Shape of cleaned data:", df.shape)
print("Missing values:")
print(df.isnull().sum())

# Check data consistency
print("Event type distribution:")
print(df['event_type'].value_counts())
print("Device type distribution:")
print(df['device_type'].value_counts())

# Export the cleaned data
df.to_csv('ecommerce_data_cleaned.csv', index=False)

# Create a summary for further analysis
session_summary = df.groupby('session_id').agg({
    'user_id': 'first',
    'date': 'first',
    'device_type': 'first',
    'view_count': 'first',
    'cart_count': 'first',
    'purchase_count': 'first',
    'session_duration': 'first',
    'has_purchase': 'first'
})

session_summary.to_csv('session_summary.csv', index=True)
```

## Results Analysis

Through the data cleaning and preprocessing process, we have addressed the following data quality issues:

1. **Missing Value Handling**: Filled missing prices, user regions, and product categories, and dropped records missing critical information.
2. **Outlier Handling**: Detected and handled price outliers, and corrected invalid timestamps and event types.
3. **Data Standardization**: Unified time formats, standardized product categories, and extracted page types and device types.
4. **Feature Engineering**: Created derived features such as session duration, activity counts, purchase flag, and price range.

The cleaned dataset is now more complete, consistent, and structured, providing a solid foundation for subsequent user behavior analysis.

## Advanced Challenges

If you have completed the basic tasks, consider trying the following advanced challenges:

1. **Advanced Missing Value Imputation**: Use machine learning models to predict missing values.
2. **User Behavior Sequence Analysis**: Analyze the sequence of user actions within a session and their conversion paths.
3. **Anomaly Detection Algorithms**: Apply algorithms like isolation forest to automatically detect multi-dimensional anomalies.
4. **Feature Importance Analysis**: Evaluate the predictive power of different features on purchase behavior.
5. **Data Visualization**: Create interactive dashboards to display insights from the cleaned data.

## Summary and Reflection

Through this project, we learned how to address common data quality issues in e-commerce user data and prepare it for further analysis. Data preprocessing is fundamental to data analysis and mining, and high-quality data is crucial for drawing reliable conclusions.

In practical applications, this kind of data cleaning work helps e-commerce platforms better understand user behavior, optimize user experience, and improve conversion rates and sales. For example, analyzing differences in conversion rates across device types can inform targeted improvements to mobile or desktop interfaces.

### Reflection Questions

1. What factors should be considered when choosing an appropriate strategy for handling missing values?
2. Should outliers always be removed or corrected? Under what circumstances might outliers contain valuable information?
3. How can the effectiveness of data cleaning and preprocessing be evaluated? What metrics can measure the improvement in data quality?

<div class="practice-link">
  <a href="/projects/preprocessing/medical-missing-values.html" class="button">Next Project: Handling Missing Values in Medical Data</a>
</div>

