import streamlit as st
import sqlite3
import re
db_url = st.secrets["db_url"]  
st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Get In Touch!</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
If you have any inquiries or would like to discuss potential projects, please fill out the contact form below.
""")
st.write('\n')
st.write('\n') 

# Initialize the SQLite database
def init_db():
    with conn.cursor() as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL
            )
        ''')
        conn.commit()

# Function to insert feedback into the SQLite database
def insert_feedback(name, email, message):
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO feedback (name, email, message) VALUES (?, ?, ?)', 
                       (name, email, message))
        conn.commit()

# Function to validate email
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

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

# Connect to the database using Streamlit's connection management
conn = st.connection('feedback_db', type='sql')

# Initialize the database
init_db()
