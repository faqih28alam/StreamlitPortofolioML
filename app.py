import streamlit as st
import numpy as np
import pandas as pd
# from sklearn.datasets import load_iris, load_digits
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image
import pickle
import time

def heart():
    st.write("""
    This app predicts **Heart Disease**
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    
    For these project, I have used Decision Tree Classifier algorithm to predict whether a person has heart disease or not based on various health parameters.         
    """)
    
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type (cp)', 0, 3, 0)
            thalach = st.sidebar.slider("Max heart rate achieved (thalach)", 71, 202, 112)
            slope = st.sidebar.slider("Slope of ST segment (slope)", 0, 2, 1)
            oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.2, 0.6)
            exang = st.sidebar.slider("Exercise induced angina (exang)", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels (ca)", 0, 3, 1)
            thal = st.sidebar.slider("Thalium test result (thal)", 1, 3, 1)
            sex = st.sidebar.selectbox("Sex", ('Female', 'Male'))
            sex = 0 if sex == "Female" else 1
            age = st.sidebar.slider("Age", 29, 77, 57)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()
    
    # Load the scaler only once, after input_df is defined
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    try:
        img = Image.open("heart-disease.jpg")
        st.image(img, width=500)
    except:
        st.info("heart-disease.jpg not found, please add the image to your folder.")
    
    if st.sidebar.button('Predict!'):
        input_scaled = scaler.transform(input_df)
        st.write(input_df)
        with open("output_decision_tree.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(input_scaled)
        result = ['No Heart Disease' if prediction[0] == 0 else 'Yes Heart Disease']
        st.subheader('Prediction:')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(2)
            st.success(f"Prediction of this app is {output}")


# Set page configuration
st.set_page_config(page_title="ML Portfolio", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Portfolio")
st.write("Welcome to my Machine Learning portfolio website! Here you can find my projects, skills, and contact information.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "ML Dashboard", "Contact"])

if page == "Home":
    st.header("About Me")
    st.write("""
        Hi, I'm FAQIH, a Machine Learning enthusiast.
        I specialize in building intelligent solutions using Python, scikit-learn, TensorFlow, and more.
    """)
    st.subheader("My Models")
    # List of image URLs (replace/add your own)
    images = [
        "Streamlit/heart-disease.jpg",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        "https://images.unsplash.com/photo-1465101046530-73398c7f28ca",
        "https://images.unsplash.com/photo-1454023492550-5696f8ff10e1",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e"
    ]

    # Display images in a 3-column grid
    cols = st.columns(3)
    for idx, img_url in enumerate(images):
        with cols[idx % 3]:
            st.image(img_url, use_container_width=True)


elif page == "ML Dashboard":
    st.header("ML Dashboard")
    dashboard_type = st.selectbox("Choose algorithm type", ["[Supervised/Unsupervised/Reinforcement Learning]","Supervised", "Unsupervised", "Reinforcement Learning"])

    if dashboard_type == "Supervised":
        st.write("I have experience with various supervised learning algorithms including Random Forest, SVM, Logistic Regression and more.")
        st.subheader("Choose Model: ")
        model = st.selectbox("Select a model", ["-","Heart Disease","Random Forest", "SVM", "Logistic Regression"])
        st.write(f"You selected: {model}")
        if model == "Random Forest":
            st.subheader("Random Forest Classifier")
            dataset = st.selectbox("Select Dataset", ["Iris", "Digits"])
            if dataset == "Iris":
                data = load_iris()
                X = data.data
                # Load trained model
                #clf = joblib.load("iris_rf_model.pkl")
                st.write("Model loaded successfully!")

                st.markdown("### Manual Input for Prediction")
                feature_values = []
                for i, feature in enumerate(data.feature_names):
                    val = st.slider(
                        label=feature,
                        min_value=float(X[:, i].min()),
                        max_value=float(X[:, i].max()),
                        value=float(X[:, i].mean()),
                        step=0.1
                    )
                    feature_values.append(val)
                if st.button("Predict"):
                    input_array = np.array(feature_values).reshape(1, -1)
                    prediction = clf.predict(input_array)[0]
                    pred_class = data.target_names[prediction]
                    st.success(f"Predicted Class: **{pred_class}**")
        
        elif model == "Heart Disease":
            heart()
        
        elif model == "SVM":
            st.subheader("Support Vector Machine")
            st.write("SVM model implementation goes here.")

        elif model == "Logistic Regression":
            st.subheader("Logistic Regression")
            st.write("Logistic Regression model implementation goes here.")

    elif dashboard_type == "Unsupervised":
        st.write("I have not been worked with unsupervised learning techniques, i will update it soon. ")
        # st.subheader("Choose Model: ")
        # model = st.selectbox("Select a model", ["-","KMeans", "DBSCAN", "Hierarchical Clustering"])
        # st.write(f"You selected: {model}")

    elif dashboard_type == "Reinforcement Learning":
        st.write("I have not been worked with reinforcement learning techniques, i will update it soon. ")
        # st.subheader("Choose Model: ")
        # model = st.selectbox("Select a model", ["-","Q-Learning", "Deep Q-Network", "Policy Gradient"])
        # st.write(f"You selected: {model}")

elif page == "Contact":
    st.header("Contact Me")
    # st.image("https://media.licdn.com/dms/image/v2/D5635AQE0W6MhTay6lw/profile-framedphoto-shrink_200_200/B56ZbPjNFpGoAc-/0/1747238834805?e=1758549600&v=beta&t=3R5c7BqFf-E2Qc_zHKYvYEPtL49NI1TE2XEG4D-Ek3U", width=200)
    st.write("Email: faqih28alam@gmail.com")
    st.write("LinkedIn: [Faqih Alam](https://www.linkedin.com/in/faqih82alam/)")
    st.write("GitHub: [Faqih28alam](https://github.com/faqih28alam)")
    st.markdown(
    """
    <div style="position:fixed; left:0; bottom:20px; width:100%; display:flex; justify-content:center; align-items:center; z-index:999;">
        <img src="https://media.licdn.com/dms/image/v2/D5635AQE0W6MhTay6lw/profile-framedphoto-shrink_200_200/B56ZbPjNFpGoAc-/0/1747238834805?e=1758549600&v=beta&t=3R5c7BqFf-E2Qc_zHKYvYEPtL49NI1TE2XEG4D-Ek3U"
             style="width:120px; height:120px; object-fit:cover; border-radius:50%; box-shadow:0 2px 8px rgba(0,0,0,0.15);" />
    </div>
    """,
    unsafe_allow_html=True
)

# ...existing model code...


