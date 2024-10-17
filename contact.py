import streamlit as st
import sqlite3
import re

st.markdown(
        f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Get In Touch!</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", unsafe_allow_html=True)
st.write('\n')
st.write("""
If you have any inquiries or would like to discuss potential projects, please fill out the contact form below.
""")
st.write('\n')
st.write('\n') 

# Initialize and set up database connection
def init_db():
    with st.connection("feedback_db", type="sql") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL
            )
        ''')
        conn.commit()

# Insert feedback into the database
def insert_feedback(name, email, message):
    with st.connection("feedback_db", type="sql") as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedback (name, email, message) VALUES (?, ?, ?)', 
                       (name, email, message))
        conn.commit()

# Email validation function
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

# Create and handle the form submission
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
            init_db()  # Ensure the database is initialized
            insert_feedback(name, email, message)
            st.success("Thank you for your interest in connecting with us!")
