import streamlit as st
import joblib
import re
import string

# 1. Text Cleaning Function (Must match your training logic)
def wordopt(text):
    text = text.lower()
    
    # Notice the 'r' before the quotes below:
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r"\W", " ", text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    return text

# 2. Load Model and Vectorizer
# Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder as this file
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: 'model.pkl' or 'vectorizer.pkl' not found. Please run your training script first.")
    st.stop()


# 3. Output Function (CORRECTED)
def output_label(n):
    if n == 1:
        return "Real News"  # Matches Training Label 1
    elif n == 0:
        return "Fake News"  # Matches Training Label 0
    else:
        return "Uncertain"

# 4. Streamlit UI Logic
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("🕵️ Fake News Detection Website")
st.markdown("---")

# Initialize Session State for the input box (Required for the Reset button to work)
if 'user_text' not in st.session_state:
    st.session_state['user_text'] = ""

# Callback function to clear text
def clear_text():
    st.session_state['user_text'] = ""

# Input Field
# Note: 'key' binds this input to st.session_state['user_text']
input_text = st.text_area("Enter the news article below to check:", 
                          height=200, 
                          key='user_text')

# Button Columns
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    # The prediction ONLY happens inside this button's if-statement
    analyze_btn = st.button("Analyze News", type="primary")

with col2:
    # This button triggers the clear_text function
    reset_btn = st.button("Reset", on_click=clear_text)

# 5. Prediction Logic
if analyze_btn:
    if input_text.strip(): # Check if text is not empty
        with st.spinner('Analyzing...'):
            # Preprocess
            testing_news = {"text": [input_text]}
            new_def_test = wordopt(testing_news["text"][0])
            new_x_test = [new_def_test]
            
            # Vectorize
            new_xv_test = vectorizer.transform(new_x_test)
            
            # Predict
            pred_svm = model.predict(new_xv_test)
            
            # Display Result
            result = output_label(pred_svm[0])
            
            st.markdown("### Result:")
            if result == "Fake News":
                st.error(f"🚨 Prediction: {result}")
            else:
                st.success(f"✅ Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some text first.")