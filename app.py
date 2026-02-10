import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np

# --- 1. Data & Model Setup ---

@st.cache_resource
def train_model():
    emails = [
        "Win a free iPhone now", "Meeting at 11 am tomorrow",
        "Congratulations you won lottery", "Project discussion with team",
        "Claim your prize immediately", "Please find the attached report",
        "Limited offer buy now", "Urgent offer expires today",
        "Schedule the meeting for Monday", "You have won a cash prize",
        "Monthly performance report attached", "Exclusive deal just for you"
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words='english', ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(emails)
    svm_model = LinearSVC(C=1.0)
    svm_model.fit(X, labels)
    
    return vectorizer, svm_model

vectorizer, svm_model = train_model()

# --- 2. Streamlit UI ---
st.title("📧 Spam Detection Assistant")
st.write("Type an email below to check if it's Spam or Ham (Safe).")

user_input = st.text_area("Enter email message:", placeholder="e.g., You won a million dollars!")

if st.button("Analyze Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Transformation and Prediction
        new_email_vector = vectorizer.transform([user_input])
        prediction = svm_model.predict(new_email_vector)
        
        # Display Result
        st.divider()
        if prediction[0] == 1:
            st.error("### Result: This looks like SPAM! ")
        else:
            st.success("### Result: This looks SAFE. ")
            
        st.info("Note: This is a basic demo model trained on 12 sample sentences.")
