import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Converts "order date (DateOrders)" column to datetime format.
df["order date (DateOrders)"]=pd.to_datetime(df["order date (DateOrders)"])
# Creates a new column "Day" based on the weekday of the order date.
df['Day'] = df["order date (DateOrders)"].dt.dayofweek
data = df[['Product Name', 'Sales', 'Days for shipping (real)','Order Item Quantity','order date (DateOrders)']].copy()
data['Date'] = pd.to_datetime(data['order date (DateOrders)'])

for product_name, product_group in data.groupby('Product Name'):
    # Calculate order item quantity for each product
    order_item_quantity = product_group['Order Item Quantity'].sum()

    # Get the last 30 historical data for each product
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
   #print(timeseries_data)


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
    y = y.reshape((-1, 1))  # reshape y to (samples, 1)

    # Building LSTM Model
    # define model
    input_layer = Input(shape=(n_steps, n_features))
    model = Sequential()
    model.add(input_layer)
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)

   # ...

    #...

    # demonstrate prediction for next 7 days
    x_input = np.array(last_30_numerical[-3:])
    temp_input = list(x_input)
    lst_output = []
    i = 0
    #print(x_input)
    while (i < 8):
        if (len(temp_input) > 3):
            x_input = np.array([temp_input[-3:]]).reshape(-1, n_features)  # Reshape to (3, n_features)
            x_input = x_input.reshape((1, 3, n_features))  # Reshape to (1, 3, n_features)
            #print("{} day input {}".format(i,x_input))
            yhat = model.predict(x_input, verbose=0)
            yhat = int(np.round(yhat[0][0]))  # Round to nearest integer
            print("{} day output {}".format(i, yhat))
            temp_input.append([yhat] * n_features)  # Append the integer value
            lst_output.append(yhat)
            i = i + 1
        else:
            x_input = np.array(temp_input).reshape(-1, n_features)  # Reshape to (len(temp_input), n_features)
            x_input = x_input.reshape((1, len(temp_input), n_features))  # Reshape to (1, len(temp_input), n_features)
            yhat = model.predict(x_input, verbose=0)
            yhat = int(np.round(yhat[0][0]))  # Round to nearest integer
            print("{} day output {}".format(i, yhat))
            temp_input.append([yhat] * n_features)  # Append the integer value
            lst_output.append(yhat)
            i = i + 1

#...

    # ...

    # Print the forecasted demand for each day
   # print("Forecasted demand for each day:", lst_output)

    # Calculate demand rate
    demand_rate = np.mean(lst_output)
    total = np.sum(lst_output)
  # Calculate lead time for each product
    lead_time = product_group['Days for shipping (real)'].mean()
    max_lead_time = product_group['Days for shipping (real)'].max()
    # Calculate safety stock
    safety_stock = (max(lst_output)*max_lead_time) - (demand_rate*lead_time)

    # Calculate reorder point
    reorder_point = demand_rate*lead_time + safety_stock



    # Print the results for this product
    print(f"Product Name: {product_name}")
    print(f"Total units sold in last 30 days for {product_name}: {last_30_data['Units Sold(Daily)'].sum()}")
    print(f"No. of units that can be sold in the next 7 days {product_name}: {total-lst_output[0]}")
    print(f"Safety Stock for {product_name}: {safety_stock}")
    print(f"Average Lead time for {product_name}: {lead_time}")
    print(f"Reorder point for {product_name}: {reorder_point}")
    print()