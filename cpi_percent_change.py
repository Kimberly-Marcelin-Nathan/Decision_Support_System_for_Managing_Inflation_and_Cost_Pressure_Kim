import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Convert 'Year' and 'Month' to datetime format and create a 'Date' column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# Define relevant categories for visualization
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
    "<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><br><i>CPI Monthly Trends User Input Parameters</i></h5>", 
    unsafe_allow_html=True
)
st.write('\n')

# User input for sector, year, and category
selected_sector = st.selectbox('Select Sector', df['Sector'].unique())
selected_year = st.selectbox('Select Year', df['Year'].unique())
selected_category = st.selectbox('Select Category', relevant_categories)

# Filter data based on user input
filtered_data = df[(df['Sector'] == selected_sector) & (df['Year'] == selected_year)]

# Create a complete date range for the selected year
date_range = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-01', freq='MS')
filtered_data = filtered_data.set_index('Date').reindex(date_range).reset_index()
filtered_data.rename(columns={'index': 'Date'}, inplace=True)

# Check if category exists in the dataset
if selected_category in filtered_data.columns:
    # Calculate the percentage change in CPI from the previous month
    filtered_data['CPI_Change'] = filtered_data[selected_category].pct_change() * 100

    # Fill NaN values in 'CPI_Change' with 0 (or another placeholder value)
    filtered_data['CPI_Change'].fillna(0, inplace=True)

    # Create combined figure
    fig = go.Figure()

    # Add Bar chart for CPI
    fig.add_trace(go.Bar(
        x=filtered_data['Date'],
        y=filtered_data[selected_category],
        name='CPI',
        marker_color='royalblue',
        texttemplate='%{y:.2f}',
        textposition='outside',
        opacity=0.75,
        yaxis='y1'
    ))

    # Add Line chart for percentage change
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['CPI_Change'],
        mode='lines+markers',
        name='Percentage Change',
        line=dict(color='darkorange', width=2),
        marker=dict(size=8, color='darkorange', symbol='circle'),
        yaxis='y2'
    ))

    st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>CPI and Percentage Change for {selected_category} in {selected_sector} Sector ({selected_year})</i></h5>", 
        unsafe_allow_html=True
    )
    st.write('\n')

    # Update layout to have two y-axes with enhanced styling
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='CPI Value',
        yaxis=dict(
            title='CPI Value',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            gridcolor='lightgrey',
            zerolinecolor='lightgrey'
        ),
        yaxis2=dict(
            title='Percentage Change (%)',
            titlefont=dict(color='darkorange'),
            tickfont=dict(color='darkorange'),
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        margin=dict(l=40, r=150, t=40, b=40),  # Adjust margins for wider legend
        width=1200,  # Set plot width to 1200 pixels
        height=600,  # Set plot height to 600 pixels
        legend=dict(
            x=1.05,  # Move legend further to the right
            y=0.99,
            traceorder='normal',
            orientation='v'
        ),
        plot_bgcolor='rgba(245,245,245,0.8)',
        paper_bgcolor='white'
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Selected category does not exist in the dataset.")
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><hr style='height: 4px;background: linear-gradient(to right, #C982EF, #b8b8b8);'><br><i>Recommendations</i></h5><br>", unsafe_allow_html=True)
st.write('\n')
if filtered_data['CPI_Change'].max() > 10:
    st.write("**Significant Increase**: The percentage change line shows a significant increase. This could indicate high inflation for this category. Businesses might want to adjust pricing strategies, while consumers should be mindful of increased costs.")
elif filtered_data['CPI_Change'].min() < -10:
    st.write("**Significant Decrease**: The percentage change line shows a significant decrease. This might indicate reduced inflation or deflation. Businesses could potentially leverage lower costs, and consumers might benefit from lower prices.")
elif filtered_data['CPI_Change'].max() < 5 and filtered_data['CPI_Change'].min() > -5:
    st.write("**Stable Trend**: The CPI values and percentage changes are relatively stable. Continue monitoring to stay updated on any future significant changes.")
else:
    st.write("**Seasonal or Anomalous Trends**: There might be seasonal patterns or anomalies in the data. Review historical data and plan accordingly.")