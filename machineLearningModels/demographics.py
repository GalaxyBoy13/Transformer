import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('drive/MyDrive/DataCoSupplyChainDataset.csv', encoding='latin-1')

grouped_data = df.groupby('Category Name')['Sales'].sum().reset_index()
fig = px.bar(grouped_data, x='Category Name', y='Sales', title='Aggregate Sales by Category')
fig.show()

Category_Name=df.groupby(['Category Name'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= True)
px.bar(Category_Name, y='Number of Orders',x = 'Category Name',color ='Number of Orders')

region_sales_per_customer = region['Sales per customer'].sum().sort_values(ascending=False).reset_index()
fig2 = px.bar(
    region_sales_per_customer,
    x='Order Region',
    y='Sales per customer',
    title="Total sales for all regions",
    labels={'Sales per customer': 'Total Sales'},
    template='plotly_dark',
    width=800,
    height=600
)
fig2.show()

# Check if 'India' is in the 'Order Country' column
if 'India' in df['Order Country'].unique():
    # Filter the data to include only the Indian market
    indian_market_data = df[df['Order Country'] == 'India']
    # Check if there are any rows in the indian_market_data
    if len(indian_market_data) > 0:
        # Group the data by Order City for the Indian market
        indian_city = indian_market_data.groupby('Order City')['Sales per customer'].sum().sort_values(ascending=False).reset_index()
        # Check if there are any groups in the indian_city
        if len(indian_city) > 0:
            # Create a bar chart for the Indian market
            fig2 = px.bar(
                indian_city,
                x='Order City',
                y='Sales per customer',
                title="Total sales for Indian market by city",
                labels={'Sales per customer': 'Total Sales'},
                template='plotly_dark',
                width=800,
                height=600
            )
            fig2.show()
        else:
            print("No groups in the indian_city")
    else:
        print("No rows in the indian_market_data")
else:
    print("'India' is not in the 'Order Country' column")

