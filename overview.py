import streamlit as st
st.markdown(
        """
       <style>
                       
            [data-testid="stMarkdownContainer"]{
                text-align :justify;
            }
            
    
            
        </style>

        
        """,
        unsafe_allow_html=True,
    )
st.write('\n')


# Project Introduction
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Introduction</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
In an ever-evolving economic landscape, understanding and managing inflation and cost pressures is crucial for making informed policy decisions. 
Inflation, defined as the rate at which the general level of prices for goods and services is rising, affects every aspect of an economy—from 
consumer purchasing power to business profitability and government policy. So, We as a dedicated team of data scientists and analysts we are passionate 
about harnessing the power of data to drive informed decisions.
Our mission is to provide actionable insights through advanced forecasting techniques and data-driven analysis.
""")
st.write('\n')
st.write('\n')

# Expertise
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Our Expertise and Value</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
         <ul>
<li><b>Data Analytics:</b>Transforming raw data into meaningful insights.
<li><b>Machine Learning:</b> Developing predictive models to anticipate future trends.
<li><b>Business Intelligence:</b> Leveraging data to enhance business strategies.
<li><b>Integrity:</b> We uphold the highest standards of honesty and transparency.
<li><b>Innovation:</b> We embrace creativity and new ideas to solve complex problems.
<li><b>Excellence:</b> We are committed to delivering high-quality results and solutions.</ul>""", unsafe_allow_html=True)
st.write('\n')
st.write('\n')

# Collab with MoSpi
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Collaboration with MoSPI</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
This project is specifically designed to cater to the needs of the Ministry of Statistics and Programme Implementation (MOSPI). We aim to provide MOSPI with advanced forecasting tools and insights to support their data-driven decision-making processes.
""")
st.write('\n')
st.write('\n')

# Consumer Price Index (CPI) Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Consumer Price Index (CPI)?</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')

st.write("""
The Consumer Price Index (CPI) is a critical economic indicator used to measure inflation. It tracks the average change over time in the prices 
paid by consumers for a basket of goods and services. This basket includes categories such as food, housing, transportation, and healthcare. 
By analyzing CPI data, we can gain insights into inflation trends and cost pressures across different sectors of the economy.
""")
st.write('\n')
st.write('\n')

# Inflation and Cost Pressure Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Inflation and Cost Pressure</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
Inflation is not just a number; it has real-world implications for individuals, businesses, and governments. High inflation can erode consumer 
purchasing power, making it more expensive for people to buy the same goods and services. For businesses, inflation can increase input costs, 
squeezing profit margins. Governments, on the other hand, need to manage inflation to maintain economic stability and ensure sustainable growth. 
This is where the concept of cost pressure comes in—understanding how inflation affects various sectors can help in anticipating and mitigating its impact.
""")
st.write('\n')
st.write('\n')

# Role of Forecasting in Policy Making Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Role of Forecasting in Policy Making</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
Forecasting future CPI values is essential for proactive economic management. By predicting inflation trends, policymakers can take preemptive actions 
to manage cost pressures and ensure economic stability. Accurate forecasting enables the Ministry of Statistics and Programme Implementation's (MoSPI) 
National Statistical Office (NSO) and Central Statistics Office (CSO) to design and implement policies that address potential economic challenges before they escalate.
""")
st.write('\n')
st.write('\n')

# Decision Support System Using ARIMA and SARIMA Models Section

st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Decision Support System Using ARIMA and SARIMA Models</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')

st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 21px;color: #3b3b3b;padding:0px'>ARIMA Model</h5>", unsafe_allow_html=True)
st.write('\n')
st.write("""
ARIMA (AutoRegressive Integrated Moving Average) is used for non-seasonal data, making it ideal for predicting inflation trends that do not exhibit clear 
seasonal patterns. It considers past values and the relationship between them to forecast future CPI values.
""")
st.write('\n')
st.write('\n')

st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 21px;color: #3b3b3b;padding:0px'>SARIMA Model</h5>", unsafe_allow_html=True)
st.write('\n')
st.write("""
SARIMA (Seasonal ARIMA) extends the ARIMA model by incorporating seasonality. This is particularly useful when dealing with data that shows regular patterns 
over time, such as seasonal fluctuations in food prices. By accounting for both trend and seasonality, SARIMA provides more accurate forecasts for seasonal data.
""")
st.write('\n')
st.write('\n')

# Benefits for MoSPI’s NSO and CSO Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Benefits for MoSPI’s NSO and CSO</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
By integrating ARIMA and SARIMA models into a decision support system, we provide MoSPI’s NSO and CSO with a powerful tool to anticipate inflationary trends. 
This system allows policymakers to:
- **Make Data-Driven Decisions:** With accurate forecasts, policymakers can base their decisions on solid data, reducing the uncertainty associated with economic planning.
- **Manage Inflation Proactively:** By predicting inflationary pressures, the government can implement measures to control inflation before it spirals out of control.
- **Tailor Policies to Sector-Specific Needs:** With detailed forecasts for different sectors, such as food, housing, and healthcare, the government can design policies that address specific areas of concern.
- **Enhance Economic Stability:** Accurate inflation forecasting helps in maintaining economic stability by preventing unexpected shocks to the economy.
""")
st.write('\n')
st.write('\n')

# Our Team Section
st.markdown(
    f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Meet Our Team</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", 
    unsafe_allow_html=True
)
st.write('\n')

# Display team members with images
col1, col2 =  st.columns(2) 

st.markdown("""
            <style>
             [data-testid="StyledFullScreenButton"]{
                     position:static;
                     display:contents;
             }""", unsafe_allow_html=True)
with col1:
    st.image('img/K.PNG', output_format='PNG')
    st.write("**Kimberly Marcelin Nathan A**")
    st.write("Lead Data Scientist")
    st.write("With expertise in predictive modeling and time series forecasting, Kim is a skilled data scientist and her proficiency in data analytics empowers businesses to make informed decisions, driving growth and operational efficiency through data-driven strategies.")

st.markdown(
        """
       <style>
            [data-testid="stImage"] {
                max-width: 50%;
            } </style> """, unsafe_allow_html=True)
with col2:
    st.image('img/s.jpg', output_format='auto')
    st.write("**Sai Manasa B**")
    st.write("Data Scientist")
    st.write("With over 4 years of experience in data science, Sai specializes in developing machine learning models and data analysis.")
    
    st.markdown(
        """
       <style>
            [data-testid="stImage"] {
                max-width: 50%;
            } </style> """, unsafe_allow_html=True)

st.write('\n')



# Conclusion Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Conclusion</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
The *Decision Support System for Managing Inflation and Cost Pressure* is not just a forecasting tool; it is a vital component in the arsenal of policymakers. 
By providing accurate and timely forecasts, this system aids in the design and implementation of policies that safeguard economic stability and promote sustainable growth. 
Through the use of sophisticated models like ARIMA and SARIMA, we empower MoSPI’s NSO and CSO to navigate the complexities of inflation and cost pressure, ensuring that 
the Indian economy remains resilient in the face of future challenges.
""")
st.write('\n')
st.write('\n')

# Conclusion Section
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Get Involved</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
We are always looking to collaborate with like-minded professionals and organizations. If you're interested in working with us or learning more about our services, please reach out through our contact page.

<br><center><b style='font-size: 23px'>Thank you for visiting our website and learning more about us!</b></center>""",unsafe_allow_html=True )