import streamlit as st
import pandas as pd
import re
from sqlalchemy import text

# Page heading
st.markdown(
    f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Get In Touch!</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", 
    unsafe_allow_html=True
)
st.write('\n')
st.write("""
If you have any inquiries or would like to discuss potential projects, please fill out the contact form below.
""")
st.write('\n')
st.write('\n')

# Initialize the SQLite database and create the table
def init_db():
    conn = st.connection("feedback_db", type="sql")
    # Using text to explicitly declare SQL command
    create_table_query = text('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL
        )
    ''')
    conn.execute(create_table_query)

# Function to insert feedback into the SQLite database
def insert_feedback(name, email, message):
    conn = st.connection("feedback_db", type="sql")
    # Insert data using pandas
    feedback_data = pd.DataFrame({"name": [name], "email": [email], "message": [message]})
    feedback_data.to_sql("feedback", con=conn, if_exists="append", index=False)

# Function to validate email
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

# Initialize the database when the app starts
init_db()

# Create the form
with st.form(key='feedback_form'):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Perform validation
        if not name:
            st.error("Name is required.")
        elif not email:
            st.error("Email is required.")
        elif not message:
            st.error("Message is required.")
        elif not validate_email(email):
            st.error("Please enter a valid email address.")
        else:
            insert_feedback(name, email, message)
            st.success("Thank you for your interest in connecting with us!")
