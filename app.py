import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Optional performance files
try:
    accuracy = joblib.load("accuracy.pkl")
    cm = joblib.load("cm.pkl")
    performance_available = True
except:
    performance_available = False

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI Fake News Detector", layout="wide")

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}
.big-title {
    font-size:50px;
    font-weight:bold;
    text-align:center;
    color:#FF4B4B;
}
.glass-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
}
.success-banner {
    background-color: #28a745; 
    padding: 10px;
    border-radius:10px;
    color:white;
    text-align:center;
    font-size:18px;
    font-weight:bold;
}
.error-banner {
    background-color: #dc3545; 
    padding: 10px;
    border-radius:10px;
    color:white;
    text-align:center;
    font-size:18px;
    font-weight:bold;
}
.sidebar .sidebar-content {
    background-color:#111;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📌 Navigation")
option = st.sidebar.selectbox("Choose Option", ["News Prediction", "Model Performance"])

# ---------------------------
# Main Title
# ---------------------------
st.markdown('<p class="big-title">📰 AI Fake News Detection System</p>', unsafe_allow_html=True)

# ---------------------------
# PAGE 1: News Prediction
# ---------------------------
if option == "News Prediction":
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### Enter News Article or URL")
            user_input = st.text_area("Paste article here", height=200, label_visibility="collapsed")
            uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
            url_input = st.text_input("Or enter a news URL")

            if uploaded_file is not None:
                user_input = uploaded_file.read().decode("utf-8")
            elif url_input.strip() != "":
                try:
                    response = requests.get(url_input)
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Extract text only
                    paragraphs = soup.find_all("p")
                    user_input = "\n".join([p.get_text() for p in paragraphs])
                except:
                    st.warning("Unable to fetch text from URL.")

        with col2:
            st.write("### Prediction Result")
            if st.button("🔍 Analyze News"):
                if user_input.strip() == "":
                    st.warning("Please provide news content.")
                else:
                    vect = vectorizer.transform([user_input])
                    prediction = model.predict(vect)[0]
                    prob = model.predict_proba(vect)[0]
                    confidence = max(prob) * 100

                    if prediction == 1:
                        st.markdown(f'<div class="success-banner">✅ REAL NEWS ({confidence:.2f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-banner">❌ FAKE NEWS ({confidence:.2f}%)</div>', unsafe_allow_html=True)

                    st.progress(int(confidence))

                    # Downloadable report
                    report_text = f"News Article:\n{user_input}\n\nPrediction: {'REAL' if prediction==1 else 'FAKE'}\nConfidence: {confidence:.2f}%"
                    st.download_button("📄 Download Prediction Report", report_text, file_name="prediction_report.txt")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE 2: Model Performance
# ---------------------------
elif option == "Model Performance":
    if not performance_available:
        st.warning("Performance files not uploaded.")
    else:
        st.write("### 📊 Model Accuracy")
        st.success(f"Accuracy: {accuracy*100:.2f}%")
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        ax.matshow(cm)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha='center', va='center')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed by AIML Student | Professional Mini Project")