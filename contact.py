import streamlit as st
import sqlite3
import re
import os

# Get the database path from Streamlit secrets
db_path = st.secrets["sqlite"]["path"]

# Function to initialize the SQLite database
def init_db():
    with sqlite3.connect(db_path) as conn:
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

# Function to insert feedback into the SQLite database
def insert_feedback(name, email, message):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedback (name, email, message) VALUES (?, ?, ?)', 
                       (name, email, message))
        conn.commit()

# Function to validate email
def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

# Initialize the database (ensure table is created)
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
