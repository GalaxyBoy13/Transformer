import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
#from sklearn.preprocessing import MinMaxScaler
# Load data
df = pd.read_csv('drive/MyDrive/DataCoSupplyChainDataset.csv', encoding='latin-1')
# Delete unwanted columns
df = df.drop(columns=['Type','Customer City',
       'Customer Country', 'Customer Email', 'Customer Fname', 'Customer Id',
       'Customer Lname', 'Customer Password', 'Customer Segment',
       'Customer State', 'Customer Street', 'Customer Zipcode','Latitude', 'Longitude', 'Market',
       'Order City', 'Order Country', 'Order Customer Id', 'Order Region', 'Order State',
       'Order Zipcode','Product Description', 'Product Image','Delivery Status',
       'Late_delivery_risk','Order Item Cardprod Id', 'Order Item Discount',
       'Order Item Discount Rate','Order Item Profit Ratio','Product Card Id','Sales per customer','Benefit per order', 'Product Status'])
# Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# Prepare data
df["order date (DateOrders)"]=pd.to_datetime(df["order date (DateOrders)"])
df['Day'] = df["order date (DateOrders)"].dt.dayofweek
data = df[['Product Name','Product Category Id', 'Sales', 'Category Id', 'Category Name','Days for shipping (real)','Order Item Quantity','order date (DateOrders)']].copy()
data['Date'] = pd.to_datetime(data['order date (DateOrders)'])
# Group the data by category
category_data = data.groupby(['Category Id', 'Category Name'])
# Create a dictionary to store the results
category_results = {}
# Loop through each category
for (category_id, category_name), category_group in category_data:
    # Initialize a list to store the products in this category
    products = []
    # Loop through each product in this category
    for product_name, product_group in category_group.groupby('Product Name'):
        # Calculate order item quantity for each product
        order_item_quantity = product_group['Order Item Quantity'].sum()
        last_30_data = product_group.nlargest(30, 'order date (DateOrders)')
        last_30_data['Units Sold(Daily)'] = last_30_data.groupby(last_30_data['Date'].dt.date)['Order Item Quantity'].transform('sum')
        # Select only the numerical columns
        last_30_numerical = last_30_data[['Units Sold(Daily)']]
        last_30_numerical = last_30_numerical.astype(np.float32)
         # preparing independent and dependent features
        def prepare_data(timeseries_data, n_features):
            X, y = [], []
            for i in range(len(timeseries_data) - n_features):
                X.append(timeseries_data[i:i + n_features])
                y.append(timeseries_data[i + n_features])
            return np.array(X), np.array(y)
        # define input sequence
        #scaler = MinMaxScaler()
        #timeseries_data = scaler.fit_transform(timeseries_data)
        timeseries_data = last_30_numerical.values.tolist()
        timeseries_data = np.array(timeseries_data).astype(np.float32)
        # choose a number of time steps
        n_steps = 3
        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)
        # Check if X has at least 2 elements in its shape tuple
        if len(X.shape) < 2:
            print(f"Error: X shape is invalid. Skipping product {product_name}")
            continue
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        y = y.reshape((-1, 1))  # reshape y to (samples, 1
        # Create and fit the LSTM network
        input_layer = Input(shape=(n_steps, n_features))
        model = Sequential()
        model.add(input_layer)
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # Fit the model
        model.fit(X,y, epochs=100, batch_size=32, verbose=1)
        # demonstrate prediction for next 7 days
        x_input = np.array(last_30_numerical[-3:])
        temp_input = list(x_input)
        lst_output = []
        i = 0
        while (i < 7):
            if (len(temp_input) > 3):
                x_input = np.array([temp_input[-3:]]).reshape(-1, n_features)
                x_input = x_input.reshape((1, 3, n_features))
                yhat = model.predict(x_input, verbose=0)
                yhat = int(np.round(yhat[0][0]))  # Round to nearest integer
                temp_input.append([yhat] * n_features)  # Append the integer value
                lst_output.append(yhat)
                i = i + 1
            else:
                x_input = np.array(temp_input).reshape(-1, n_features)
                x_input = x_input.reshape((1, len(temp_input), n_features))
                yhat = model.predict(x_input, verbose=0)
                yhat = int(np.round(yhat[0][0]))  # Round to nearest integer
                temp_input.append([yhat] * n_features)  # Append the integer value
                lst_output.append(yhat)
                i = i + 1
        # Calculate demand rate
        demand_rate = np.mean(lst_output)
        total = np.sum(lst_output)
        # Add the product to the list
        products.append({'Product Category Id': product_group['Product Category Id'].iloc[0],
                         'Product Name': product_name,
                         'Demand Rate': demand_rate})
    # Sort the products by demand rate in descending order
    products.sort(key=lambda x: x['Demand Rate'], reverse=True)
    print(f"Category {category_id}: {category_name}")
    for product in products:
        print(f"Product Name: {product['Product Name']}, Demand Rate: {product['Demand Rate']}")
    print(f"Average Demand Rate for Category {category_id}: {np.mean([product['Demand Rate'] for product in products]):.2f}")
    print()
    print('*******************************************************************************')
    # Add the category to the results dictionary
    category_results[(category_id, category_name)] = products


import plotly.graph_objects as go

# Create a list of category names and average demand rates
category_names = [category_name for category_id, category_name in category_results.keys()]
avg_demand_rates = [np.mean([product['Demand Rate'] for product in products]) for products in category_results.values()]

# Create the plot
fig = go.Figure(data=[go.Bar(x=category_names, y=avg_demand_rates)])
fig.update_layout(title='In the next 7 days',
                  xaxis_title='Category Name',
                  yaxis_title='Average Demand Rate')

# Show the plot
fig.show()