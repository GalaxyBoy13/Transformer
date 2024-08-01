import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load data
df = pd.read_csv('drive/MyDrive/DataCoSupplyChainDataset.csv', encoding='latin-1')

# Drop unwanted columns
columns_to_drop = [
    'Type', 'Customer City', 'Customer Country', 'Customer Email',
    'Customer Fname', 'Customer Id', 'Customer Lname', 'Customer Password',
    'Customer Segment', 'Customer State', 'Customer Street', 'Customer Zipcode',
    'Latitude', 'Longitude', 'Market', 'Order City', 'Order Country',
    'Order Customer Id', 'Order Region', 'Order State', 'Order Zipcode',
    'Product Description', 'Product Image', 'Delivery Status',
    'Late_delivery_risk', 'Order Item Cardprod Id', 'Order Item Discount',
    'Order Item Discount Rate', 'Order Item Profit Ratio', 'Product Card Id',
    'Sales per customer', 'Benefit per order', 'Product Status'
]
df.drop(columns=columns_to_drop, inplace=True)

# Handling missing values
df.dropna(inplace=True)

# Removing duplicates
df.drop_duplicates(inplace=True)

# Convert columns to appropriate data types
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

if 'Shipping Date' in df.columns:
    df['Shipping Date'] = pd.to_datetime(df['Shipping Date'], errors='coerce')

# Ensure numerical columns are correctly formatted
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Handling outliers for 'Order Item Quantity' (if applicable)
if 'Order Item Quantity' in df.columns:
    upper_limit = df['Order Item Quantity'].quantile(0.99)
    df['Order Item Quantity'] = np.where(df['Order Item Quantity'] > upper_limit, upper_limit, df['Order Item Quantity'])

# Market Basket Analysis Preparation
# Group products by 'Order Id' and create a one-hot encoded dataframe
basket = df.groupby(['Order Id', 'Product Name']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Market Basket Analysis - FP-Growth Algorithm
# Using FP-Growth for more efficient handling of larger datasets
frequent_itemsets = fpgrowth(basket, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Formatting the results
formatted_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
formatted_rules = formatted_rules.round({'support': 4, 'confidence': 4, 'lift': 4})

# Calculate a composite score for sorting
# Adjust the weights as needed to balance confidence and lift
weights = {'confidence': 0.5, 'lift': 0.5}
formatted_rules['composite_score'] = (weights['confidence'] * formatted_rules['confidence'] +
                                      weights['lift'] * formatted_rules['lift'])

# Sort by composite score in descending order
sorted_by_composite = formatted_rules.sort_values(by='composite_score', ascending=False)
sorted_by_composite.to_csv('sorted_by_composite.csv', index=False)

# Display the sorted results
print("\nSorted by Composite Score (Confidence & Lift):")
print(sorted_by_composite.to_string(index=False))
