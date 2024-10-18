import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.callbacks import EarlyStopping

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
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>User Input Parameters</i></h5>", unsafe_allow_html=True)
st.write('\n')
# User input for sector and category
col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox('Select Sector', df['Sector'].unique())
    
with col2:
    category = st.selectbox('Select Category', relevant_categories)

# Filter data based on user input
filtered_data = df[df['Sector'] == sector]
time_series = filtered_data[['Date', category]].set_index('Date').asfreq('MS')[category].dropna()

# Split data into training and validation sets for LSTM
train_data_lstm = time_series[time_series.index.year <= 2022]
test_data_lstm = time_series[time_series.index.year > 2022]

# Split data into training and validation sets for ARIMA/SARIMA
train_data_arima = time_series[time_series.index.year < 2023]
validation_data_arima = time_series[time_series.index.year >= 2023]

# Prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

if st.checkbox('Update Forecasts', value=True):
    # LSTM Model
    # Scale the training data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data_lstm.values.reshape(-1, 1))

    # Prepare dataset for LSTM
    n_steps = 5  # Number of past time steps to use
    X_train, y_train = prepare_data(scaled_train_data, n_steps)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build a more complex LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(50, activation='tanh'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Add Early Stopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Fit the model
    model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping])

    # Make predictions for 2023 and 2024
    n_forecast_periods = len(test_data_lstm)
    forecast_lstm = []
    last_data = scaled_train_data[-n_steps:]

    for _ in range(n_forecast_periods):
        input_data = last_data.reshape((1, n_steps, 1))
        prediction = model_lstm.predict(input_data, verbose=0)
        forecast_lstm.append(prediction[0, 0])
        last_data = np.append(last_data[1:], prediction)

    # Inverse transform the predictions
    forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1, 1))
    future_dates_lstm = pd.date_range(start=test_data_lstm.index[0], periods=n_forecast_periods, freq='MS')

    # ARIMA Model
    model_arima = pm.auto_arima(train_data_arima, seasonal=False, stepwise=True)
    forecast_arima, conf_int_arima = model_arima.predict(n_periods=len(validation_data_arima), return_conf_int=True)

    # SARIMA Model
    model_sarima = SARIMAX(train_data_arima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = model_fit_sarima.predict(start=len(train_data_arima), end=len(train_data_arima) + len(validation_data_arima) - 1, dynamic=False)

    # Visualization
    # LSTM Plot
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=train_data_lstm.index, y=train_data_lstm, mode='lines', name='Training Data', line=dict(color='blue')))
    fig_lstm.add_trace(go.Scatter(x=test_data_lstm.index, y=test_data_lstm, mode='lines', name='Actual Test Data', line=dict(color='green')))
    fig_lstm.add_trace(go.Scatter(x=future_dates_lstm, y=forecast_lstm.flatten(), mode='lines', name='LSTM Forecast', line=dict(color='orange')))
        
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>LSTM Validation Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(fig_lstm)


    # ARIMA Validation Plot
    validation_fig_arima = go.Figure()
    validation_fig_arima.add_trace(go.Scatter(x=train_data_arima.index, y=train_data_arima, mode='lines', name='Train Data', line=dict(color='blue')))
    validation_fig_arima.add_trace(go.Scatter(x=validation_data_arima.index, y=validation_data_arima, mode='lines', name='Validation Data', line=dict(color='green')))
    validation_fig_arima.add_trace(go.Scatter(x=validation_data_arima.index, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    validation_fig_arima.update_layout(xaxis_title='Date', yaxis_title=category)
    
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>ARIMA Validation Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(validation_fig_arima)

    # SARIMA Validation Plot
    validation_fig_sarima = go.Figure()
    validation_fig_sarima.add_trace(go.Scatter(x=train_data_arima.index, y=train_data_arima, mode='lines', name='Train Data', line=dict(color='blue')))
    validation_fig_sarima.add_trace(go.Scatter(x=validation_data_arima.index, y=validation_data_arima, mode='lines', name='Validation Data', line=dict(color='green')))
    validation_fig_sarima.add_trace(go.Scatter(x=validation_data_arima.index, y=forecast_sarima, mode='lines', name='SARIMA Forecast', line=dict(color='orange')))
    validation_fig_sarima.update_layout(xaxis_title='Date', yaxis_title=category)
    
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>SARIMA Validation Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(validation_fig_sarima)
    
    
    # Comparison Plot
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=validation_data_arima.index, y=validation_data_arima, mode='lines', name='Validation Data', line=dict(color='green')))
    comparison_fig.add_trace(go.Scatter(x=future_dates_lstm, y=forecast_lstm.flatten(), mode='lines', name='LSTM Forecast', line=dict(color='blue')))
    comparison_fig.add_trace(go.Scatter(x=validation_data_arima.index, y=forecast_arima, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
    comparison_fig.add_trace(go.Scatter(x=validation_data_arima.index, y=forecast_sarima, mode='lines', name='SARIMA Forecast', line=dict(color='red')))
    comparison_fig.update_layout(xaxis_title='Date', yaxis_title=category)
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Comparison of ARIMA, SARIMA & LSTM Forecasts for {category}</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(comparison_fig)

    # Calculate performance metrics
    mae_lstm = mean_absolute_error(test_data_lstm, forecast_lstm.flatten())
    rmse_lstm = mean_squared_error(test_data_lstm, forecast_lstm.flatten(), squared=False)
    mse_lstm = mean_squared_error(test_data_lstm, forecast_lstm.flatten())
    mape_lstm = np.mean(np.abs((test_data_lstm - forecast_lstm.flatten()) / test_data_lstm)) * 100

    mae_arima = mean_absolute_error(validation_data_arima, forecast_arima)
    rmse_arima = mean_squared_error(validation_data_arima, forecast_arima, squared=False)
    mse_arima = mean_squared_error(validation_data_arima, forecast_arima)
    mape_arima = np.mean(np.abs((validation_data_arima - forecast_arima) / validation_data_arima)) * 100

    mae_sarima = mean_absolute_error(validation_data_arima, forecast_sarima)
    rmse_sarima = mean_squared_error(validation_data_arima, forecast_sarima, squared=False)
    mse_sarima = mean_squared_error(validation_data_arima, forecast_sarima)
    mape_sarima = np.mean(np.abs((validation_data_arima - forecast_sarima) / validation_data_arima)) * 100

    # Create a summary table for metrics
    metrics_df = pd.DataFrame({
    'Model': ['LSTM', 'ARIMA', 'SARIMA'],
    'Mean Absolute Error (MAE)': [mae_lstm, mae_arima, mae_sarima],
    'Root Mean Squared Error (RMSE)': [rmse_lstm, rmse_arima, rmse_sarima],
    'Mean Squared Error (MSE)': [mse_lstm, mse_arima, mse_sarima],
    'Mean Absolute Percentage Error (MAPE)': [mape_lstm, mape_arima, mape_sarima]
     })

    # Display the metrics as a table
    st.markdown(
    f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Metrics</i><br><br></h5>", 
    unsafe_allow_html=True)
    st.write('\n')
    st.dataframe(metrics_df, use_container_width=True)

    # Combine actual and predicted values for all models
    results_df = pd.DataFrame({
        'Date': validation_data_arima.index,
        'Actual': validation_data_arima.values,
        'LSTM Predicted': forecast_lstm.flatten(),
        'ARIMA Predicted': forecast_arima,
        'SARIMA Predicted': forecast_sarima
    })

    # Display the results as a table
    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Actual Vs. Predicted Table Values</i><br><br></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.dataframe(results_df, use_container_width=True)


# Create a bar chart for model performance comparison
# Create a bar chart for model performance comparison
performance_metrics = metrics_df.melt(id_vars='Model', 
                                        value_vars=['Mean Absolute Error (MAE)', 
                                                    'Root Mean Squared Error (RMSE)', 
                                                    'Mean Squared Error (MSE)', 
                                                    'Mean Absolute Percentage Error (MAPE)'])
performance_metrics.columns = ['Model', 'Metric', 'Value']

# Plot the bar chart
fig_performance = px.bar(performance_metrics, 
                          x='Model', 
                          y='Value', 
                          color='Metric', 
                          barmode='group', 
                          title='Model Performance Comparison',
                          labels={'Value': 'Error Value', 'Model': 'Model'})

# Display the bar chart in Streamlit
st.markdown(
    f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Comparison Bar Graph</i><br></h5>", 
    unsafe_allow_html=True)
st.plotly_chart(fig_performance)