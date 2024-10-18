import streamlit as st
import requests

# Set up Mailchimp API details
API_KEY = '692dbd2ef3fcffca57da0e365c4b10be-us10'
LIST_ID = '0504edf4de'
MAILCHIMP_DATA_CENTER = 'us10'  # replace with your data center, e.g., us1, us2, etc.

def subscribe_to_mailchimp(email, first_name, phone, message):
    url = f'https://{MAILCHIMP_DATA_CENTER}.api.mailchimp.com/3.0/lists/{LIST_ID}/members'
    data = {
        'email_address': email,
        'status': 'subscribed',
        'merge_fields': {
            'FNAME': first_name,
            'PHONE': phone,
            'MMERGE7': message  # Adjust the key for your Mailchimp merge field
        }
    }
    response = requests.post(url, json=data, auth=('anystring', API_KEY))
    return response.status_code, response.json()

# Display title and description
st.markdown(
    """
    <h5 style='text-align: left; letter-spacing:1px; font-size: 23px; color: #3b3b3b; padding:0px'><i>Get In Touch!</i></h5>
    <hr style='margin-top:15px; margin-bottom:10px'>
    """, unsafe_allow_html=True
)

st.write("""
If you have any inquiries or would like to discuss potential projects, please fill out the contact form below.
""")

# Create the form
with st.form(key='contact_form'):
    email = st.text_input("Email Address")
    first_name = st.text_input("Name")
    phone = st.text_input("Phone Number")
    message = st.text_area("Message")
    
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        if email and first_name and message:  # Validate required fields
            status_code, response_data = subscribe_to_mailchimp(email, first_name, phone, message)
            if status_code == 200:
                st.success("Thank you for your message! We will get back to you soon.")
            else:
                st.error(f"An error occurred: {response_data.get('detail', 'Please try again later.')}")
        else:
            st.warning("Please fill in all required fields.")
