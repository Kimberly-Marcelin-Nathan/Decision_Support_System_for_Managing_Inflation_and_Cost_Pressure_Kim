import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

# Set page configuration
st.set_page_config(layout="wide")


# Since we are not using sections, we will use pages.toml
nav = get_nav_from_toml(".streamlit/pages.toml")

# Display logo
st.logo("logo.png")

st.markdown(
        """
       <style>
            [data-testid="stLogo"] {
               content="icon";
               Height: 3rem;
            }
            
            [data-testid="baseButton-headerNoPadding"]{
                Margin-top:15px;
            }
            
            [data-testid="stHeader"]{
                Height:5.5rem;
                Background: #f0f2f6;
                position: relative;
            }
            
            [data-testid="stHeader"]::before {
                content: "Decision Support System For Managing Inflation & Cost Pressure";
                font-size: 25px;
                font-family: "Source Sans Pro", sans-serif;
                letter-spacing: 0.005em;
                color: #3b3b3b;
                font-weight: 600;
                
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                
            }
            [data-testid="stMainMenu"]{
                Margin-top:-15px;
            }
            
            [data-testid="stToolbar"]{
                Top :30px;
            }
            
            [data-testid="stAppViewBlockContainer"]{
                padding-left:7rem;
                padding-right:7rem;
            }
            
            [data-testid="stSidebar"]{
                width :270px !important;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f0f2f6;
                text-align: center;
                padding: 10px 0;
                font-size: 12px;
                color: #6c757d;
                z-index:1;
            }
    
    
            
        </style>

        <div class="footer">
        © 2024 Decision Support System By Kim & Sai
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize and run the navigation
pg = st.navigation(nav)
#add_page_title(pg)
pg.run()
