import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os

# Set the page config first, before anything else
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered",
)

# Set the path to your local nltk_data directory
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Skip downloading 'punkt' and 'stopwords' during runtime, as they are already in the 'nltk_data' directory
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.error("NLTK 'punkt' tokenizer not found. Please check your nltk_data directory.")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.error("NLTK 'stopwords' not found. Please check your nltk_data directory.")

# Function to preprocess the text
def transform_text(text):
    # Converting to lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    text = [i for i in text if i.isalnum()]

    # Removing stop words and punctuation characters
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    ps = nltk.PorterStemmer()
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the TF-IDF vectorizer and model
with open('vectorizer (1).pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #16C47F;
    }
    .header {
        color: #3E64FF;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        font-size: 40px;
    }
    .subheader {
        color: #252525;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-top: -20px;
        font-size: 18px;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .spam {
        background-color: #E78F81
    }
    .not-spam {
        background-color: #FF9D23;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title and subtitle
st.markdown('<div class="header">SMS Spam Detection</div>', unsafe_allow_html=True)

# Input from the user
st.write("### 📧 Enter your message:")
input_sms = st.text_area("Type your message here", height=150)

# Predict button
if st.button("🚀 Predict"):
    if input_sms.strip():
        # 1. Preprocess the input
        transform_sms = transform_text(input_sms)

        # 2. Vectorize the input
        vector_input = tfidf.transform([transform_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.markdown(
                '<div class="result spam">🚨 This message is classified as: SPAM</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result not-spam">✅ This message is classified as: NOT SPAM</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Please enter a message before clicking Predict.")
