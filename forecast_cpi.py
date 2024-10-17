import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

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


st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>User Input Parameters</i></h5>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox('Select Sector', df['Sector'].unique())
with col2:
    category = st.selectbox('Select Category', relevant_categories)

n_periods = st.slider('Select number of months to forecast (1-36)', 1, 36)

# Filter data based on user input
filtered_data = df[df['Sector'] == sector]
time_series = filtered_data[['Date', category]].set_index('Date').asfreq('MS')[category].dropna()

# LSTM Forecasting
st.markdown(f"<hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'>", unsafe_allow_html=True)
st.write('\n')
if st.checkbox('Use LSTM Forecast', value=True):
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>LSTM Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)    
    st.write('\n')
    # Prepare data for LSTM
    def prepare_data(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))
    n_steps = 5
    X, y = prepare_data(scaled_data, n_steps)
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

    forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1, 1))
    future_dates_lstm = pd.date_range(start=time_series.index[-1], periods=n_periods + 1, freq='MS')[1:]

    # LSTM Plot
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Actual', line=dict(color='blue')))
    fig_lstm.add_trace(go.Scatter(x=future_dates_lstm, y=forecast_lstm.flatten(), mode='lines', name='Predicted', line=dict(color='orange')))
    fig_lstm.update_layout(xaxis_title='Date', yaxis_title=category)
    
    st.plotly_chart(fig_lstm)

    # Performance Metrics
    if len(forecast_lstm) == n_periods:
        mae_lstm = mean_absolute_error(time_series[-n_periods:], forecast_lstm.flatten())
        rmse_lstm = mean_squared_error(time_series[-n_periods:], forecast_lstm.flatten(), squared=False)
        mape_lstm = mean_absolute_percentage_error(time_series[-n_periods:], forecast_lstm.flatten()) * 100

# ARIMA Forecasting
st.markdown(f"<hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'>", unsafe_allow_html=True)
st.write('\n')

if st.checkbox('Use ARIMA/SARIMA Forecast', value=True):
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>ARIMA Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n') 

    # ARIMA Model
    model_arima = pm.auto_arima(time_series, seasonal=False, stepwise=True)
    forecast_arima, conf_int_arima = model_arima.predict(n_periods=n_periods, return_conf_int=True)
    
    future_dates_arima = pd.date_range(start=time_series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    
    # ARIMA Plot
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Observed', line=dict(color='blue')))
    fig_arima.add_trace(go.Scatter(x=future_dates_arima, y=forecast_arima, mode='lines', name='Forecast', line=dict(color='orange')))
    fig_arima.add_trace(go.Scatter(
        x=list(future_dates_arima) + list(reversed(future_dates_arima)),
        y=list(conf_int_arima[:, 0]) + list(reversed(conf_int_arima[:, 1])),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    fig_arima.update_layout(xaxis_title='Date', yaxis_title=category)
       
    st.plotly_chart(fig_arima)

    # SARIMA Model
    model_sarima = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit_sarima = model_sarima.fit(disp=False)
    forecast_sarima = model_fit_sarima.predict(start=len(time_series), end=len(time_series) + n_periods - 1)

    future_dates_sarima = pd.date_range(start=time_series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    
    # SARIMA Plot
    fig_sarima = go.Figure()
    fig_sarima.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Observed', line=dict(color='blue')))
    fig_sarima.add_trace(go.Scatter(x=future_dates_sarima, y=forecast_sarima, mode='lines', name='Forecast', line=dict(color='orange')))
    fig_sarima.update_layout(xaxis_title='Date', yaxis_title=category)
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>SARIMA Forecast for {category} in {sector} Sector</i></h5>", unsafe_allow_html=True)
    st.write('\n')
    st.plotly_chart(fig_sarima)
    
    
    # Performance Metrics
if len(forecast_arima) == len(forecast_sarima) == n_periods:
    # ARIMA Metrics
    mse_arima = mean_squared_error(time_series[-n_periods:], forecast_arima)
    mae_arima = mean_absolute_error(time_series[-n_periods:], forecast_arima)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = mean_absolute_percentage_error(time_series[-n_periods:], forecast_arima) * 100
    
    # SARIMA Metrics
    mse_sarima = mean_squared_error(time_series[-n_periods:], forecast_sarima)
    mae_sarima = mean_absolute_error(time_series[-n_periods:], forecast_sarima)
    rmse_sarima = np.sqrt(mse_sarima)
    mape_sarima = mean_absolute_percentage_error(time_series[-n_periods:], forecast_sarima) * 100

    # LSTM Metrics
    mse_lstm = mean_squared_error(time_series[-n_periods:], forecast_lstm.flatten())
    mae_lstm = mean_absolute_error(time_series[-n_periods:], forecast_lstm.flatten())
    rmse_lstm = np.sqrt(mse_lstm)
    mape_lstm = mean_absolute_percentage_error(time_series[-n_periods:], forecast_lstm.flatten()) * 100

    # Comparison DataFrame
    performance_metrics = pd.DataFrame({
        'Model': ['ARIMA', 'SARIMA', 'LSTM'],
        'Mean Absolute Error (MAE)': [mae_arima, mae_sarima, mae_lstm],
        'Mean Squared Error (MSE)': [mse_arima, mse_sarima, mse_lstm],
        'Root Mean Squared Error (RMSE)': [rmse_arima, rmse_sarima, rmse_lstm],
        'Mean Absolute Percentage Error (MAPE)': [mape_arima, mape_sarima, mape_lstm]
    })
    st.markdown(
            f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Metrics</i></h5><br><br>", 
            unsafe_allow_html=True
        )
    st.write('\n')
    st.dataframe(performance_metrics, use_container_width=True)

    # Comparison Plot
    comparison_fig = go.Figure()
    for metric in performance_metrics.columns[1:]:
        comparison_fig.add_trace(go.Bar(x=performance_metrics['Model'], y=performance_metrics[metric], name=metric))
    comparison_fig.update_layout(barmode='group', title='Model Performance Comparison', xaxis_title='Model', yaxis_title='Error Metric')
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Model Performance Comparison (MAE, RMSE, MSE, MAPE)</i></h5>", unsafe_allow_html=True)
    st.plotly_chart(comparison_fig)
    
    best_model = None
    best_metrics = {}

    if mae_lstm < mae_arima and mae_lstm < mae_sarima:
       best_model = "LSTM"
       best_metrics = {
               'MAE': mae_lstm,
               'RMSE': rmse_lstm,
               'MSE': mse_lstm,
               'MAPE': mape_lstm
        }
    elif mae_arima < mae_sarima and rmse_arima < rmse_sarima:
       best_model = "ARIMA"
       best_metrics = {
               'MAE': mae_arima,
               'RMSE': rmse_arima,
               'MSE': mse_arima,
               'MAPE': mape_arima
        }
    else:
      best_model = "SARIMA"
      best_metrics = {
               'MAE': mae_sarima,
               'RMSE': rmse_sarima,
               'MSE': mse_sarima,
               'MAPE': mape_sarima
        }

        # Recommendation section
    st.markdown(f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Recommended Model</i></h5><br>", unsafe_allow_html=True)
    st.markdown(f"<h6 style='text-align: left; letter-spacing:1px;font-size: 18px; font-weight: 250;color: #3b3b3b;padding:0px'><br>The recommended model for forecasting CPI for {category} in {sector} sector is <b>{best_model}</b></h6><br>", unsafe_allow_html=True)
    for metric, value in best_metrics.items():
       st.write(f"**Best {metric}:** {value:.4f}")