import streamlit as st

st.markdown(
    f"<h5 style='text-align: left; letter-spacing:1px;font-size: 23px;color: #3b3b3b;padding:0px'><i>Get In Touch!</i></h5><hr style='margin-top:15px; margin-bottom:10px'>", 
    unsafe_allow_html=True
)
st.write('\n')
st.write("""
If you have any inquiries or would like to discuss potential projects, please fill out the contact form below.
""")

# Mailchimp form embedded as HTML with responsive CSS
mailchimp_form = '''
<div id="mc_embed_shell">
      <link href="//cdn-images.mailchimp.com/embedcode/classic-061523.css" rel="stylesheet" type="text/css">
  <style type="text/css">
        #mc_embed_signup {
            background:#fff;
            clear:left;
            font:14px Helvetica,Arial,sans-serif;
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }
        #mc_embed_signup #mce-success-response {
            color: #bb4cf6;
            display: none;
        }
        /* Adjust input field width for mobile */
        #mc_embed_signup input {
            width: 100%;
            height: 50px;
            box-sizing: border-box;
        }
        #mc_embed_signup_scroll {
            padding: 10px;
        }
        #mc_embed_signup .button:hover {
            background-color: #c982ef;
        }
        /* Make the submit button responsive */
        #mc_embed_signup .button {
            width: 50%;
            padding: 10px;
            font-size: 16px;
            background-color: #bb4cf6;
            color: white;
            height: 52px;
            border: none;
            cursor: pointer;
        }
        #mc_embed_signup div#mce-responses {
            margin: 10px 5%;
        }    
        /* Add some responsiveness */
        @media only screen and (max-width: 600px) {
            #mc_embed_signup {
                padding: 0 20px;
            }
            #mc_embed_signup h2 {
                font-size: 20px;
            }
        }
</style>
<div id="mc_embed_signup">
    <form action="https://jimkimproduction.us10.list-manage.com/subscribe/post?u=2c14052d630b9197b52a3dede&amp;id=0504edf4de&amp;f_id=00844be4f0" method="post" id="mc-embedded-subscribe-form" name="mc-embedded-subscribe-form" class="validate" target="_blank">
        <div id="mc_embed_signup_scroll">
            <h2>Talk to us!</h2>
            <div class="indicates-required"><span class="asterisk">*</span> indicates required</div>
            <div class="mc-field-group">
                <label for="mce-EMAIL">Email Address <span class="asterisk">*</span></label>
                <input type="email" name="EMAIL" class="required email" id="mce-EMAIL" required="" value="">
            </div>
            <div class="mc-field-group">
                <label for="mce-FNAME">Name <span class="asterisk">*</span></label>
                <input type="text" name="FNAME" class="required text" id="mce-FNAME" required="" value="">
            </div>
            <div class="mc-field-group">
                <label for="mce-PHONE">Phone Number</label>
                <input type="text" name="PHONE" class="REQ_CSS" id="mce-PHONE" value="">
            </div>
            <div class="mc-field-group">
                <label for="mce-MMERGE7">Message <span class="asterisk">*</span></label>
                <input type="text" name="MMERGE7" class="required text" id="mce-MMERGE7" required="" value="">
            </div>
        <div id="mce-responses" class="clear foot">
            <div class="response" id="mce-error-response" style="display:none;"></div>
            <div class="response" id="mce-success-response" style="display:none;"></div>
        </div>
        <div aria-hidden="true" style="position:absolute; left:-5000px;">
            <input type="text" name="b_2c14052d630b9197b52a3dede_0504edf4de" tabindex="-1" value="">
        </div>
        <div class="optionalParent">
            <div class="clear foot">
                <input type="submit" name="subscribe" id="mc-embedded-subscribe" class="button" value="Submit">
            </div>
        </div>
    </div>
    </form>
</div>
<script type="text/javascript" src="//s3.amazonaws.com/downloads.mailchimp.com/js/mc-validate.js"></script>
<script type="text/javascript">(function($) {window.fnames = new Array(); window.ftypes = new Array();fnames[0]='EMAIL';ftypes[0]='email';fnames[1]='FNAME';ftypes[1]='text';fnames[4]='PHONE';ftypes[4]='phone';fnames[7]='MMERGE7';ftypes[7]='text';}(jQuery));var $mcj = jQuery.noConflict(true);</script>
'''

# Render the form using Streamlit's `st.components.v1.html` function
st.components.v1.html(mailchimp_form, height=630)
