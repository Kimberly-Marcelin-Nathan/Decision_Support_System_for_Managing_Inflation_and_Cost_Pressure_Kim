import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Convert 'Year' and 'Month' to datetime format and create a 'Date' column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# Define relevant categories for forecasting
relevant_categories = [
    'Cereals and products', 'Meat and fish', 'Egg', 'Milk and products',
    'Oils and fats', 'Fruits', 'Vegetables', 'Pulses and products',
    'Sugar and Confectionery', 'Spices', 'Non-alcoholic beverages',
    'Prepared meals, snacks, sweets etc.', 'Food and beverages',
    'Pan, tobacco and intoxicants', 'Clothing', 'Footwear',
    'Clothing and footwear', 'Fuel and light',
    'Household goods and services', 'Health',
    'Transport and communication', 'Recreation and amusement',
    'Education', 'Personal care and effects', 'Miscellaneous', 'General index'
]

# Streamlit app title
st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>User Input Parameters</i></h5>", unsafe_allow_html=True)
st.write('\n')

# Create a two-column layout for user input and results
col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox('Select Sector', df['Sector'].unique())
    
with col2:
    category = st.selectbox('Select Category', relevant_categories)

n_periods = st.slider('Select number of months to forecast (1-120)', 1, 120)


# Filter data based on user input
filtered_data = df[df['Sector'] == sector]
time_series = filtered_data[['Date', category]].set_index('Date').asfreq('MS')[category].dropna()

# Prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

if st.checkbox('Update LSTM Forecast', value=True):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

    # Prepare dataset
    n_steps = 5  # Number of past time steps to use
    X, y = prepare_data(scaled_data, n_steps)

    # Reshape input for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, verbose=0)

    # Make predictions
    forecast_lstm = []
    last_data = scaled_data[-n_steps:]

    for _ in range(n_periods):
        input_data = last_data.reshape((1, n_steps, 1))
        prediction = model.predict(input_data, verbose=0)
        forecast_lstm.append(prediction[0, 0])
        last_data = np.append(last_data[1:], prediction)

    # Inverse transform the predictions
    forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1, 1))

    # Create a date range for the forecasted values
    future_dates_lstm = pd.date_range(start=time_series.index[-1], periods=n_periods + 1, freq='MS')[1:]

    # LSTM Plot
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Actual', line=dict(color='blue')))
    fig_lstm.add_trace(go.Scatter(x=future_dates_lstm, y=forecast_lstm.flatten(), mode='lines', name='Predicted', line=dict(color='orange')))

    # Add layout details
    fig_lstm.update_layout(
        
        xaxis_title='Date',
        yaxis_title=category,
        legend=dict(x=0, y=1)
    )
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>LSTM Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(fig_lstm)

    # Calculate performance metrics
    if len(forecast_lstm) == n_periods:
        mae_lstm = mean_absolute_error(time_series[-n_periods:], forecast_lstm.flatten())
        rmse_lstm = mean_squared_error(time_series[-n_periods:], forecast_lstm.flatten(), squared=False)
        mse_lstm = mean_squared_error(time_series[-n_periods:], forecast_lstm.flatten())
        
        # Calculate MAPE
        mape_lstm = np.mean(np.abs((time_series[-n_periods:] - forecast_lstm.flatten()) / time_series[-n_periods:])) * 100

        st.markdown(
            f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Metrics</i></h5><br>", 
            unsafe_allow_html=True
        )
        st.write('\n')
        st.write(f"**Mean Absolute Error (MAE):** {mae_lstm:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse_lstm:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse_lstm:.4f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape_lstm:.2f}%")

        # Create a bar plot for performance metrics
        metrics = {
            'MAE': mae_lstm,
            'RMSE': rmse_lstm,
            'MSE': mse_lstm,
            'MAPE': mape_lstm
        }

        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color='indigo'))

        fig_metrics.update_layout(
            
            xaxis_title='Metrics',
            yaxis_title='Values',
            yaxis=dict(range=[0, max(metrics.values()) * 1.1])
        )
        st.markdown(
            f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Metrics Plot</i></h5><br>", 
            unsafe_allow_html=True
        )
        st.write('\n')
        st.plotly_chart(fig_metrics)