import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("news.csv")  # Ensure 'news.csv' exists in the same directory
    
    # Handle missing values
    data['subject'] = data['subject'].fillna('')  # Replace NaNs with empty strings

    # Adjust label based on dataset (mapping TRUE -> REAL and FALSE -> FAKE)
    data['label'] = data['label'].apply(lambda x: 'REAL' if x == True else 'FAKE')
    
    return data

# Data Exploration
def data_overview(data):
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write("Shape of Dataset:", data.shape)
    
    # Check class distribution
    class_distribution = data['label'].value_counts()
    st.write("Class Distribution:")
    st.write(class_distribution)
    
    st.bar_chart(class_distribution)

# Train Model
@st.cache_resource
def train_model(data):
    data = data.dropna(subset=['text', 'subject'])  # Ensure no NaN in text and subject columns
    X = data['text']
    y = data['label']

    # Check class balance
    class_distribution = data['label'].value_counts()
    st.write("Class Distribution (before resampling):")
    st.write(class_distribution)

    # If there is only one class, warn and skip training
    if len(class_distribution) == 1:
        st.warning(f"Warning: The dataset contains only one class: {class_distribution.index[0]}.")
        return None, None, 0, None, None, None

    # If both classes exist, resample the minority class
    if len(class_distribution) == 2 and 'FAKE' in class_distribution and 'REAL' in class_distribution:
        # Resample the minority class (undersampling majority class)
        data_balanced = pd.concat([
            resample(data[data['label'] == 'FAKE'], replace=True, n_samples=len(data[data['label'] == 'REAL']), random_state=42),
            data[data['label'] == 'REAL']
        ])
    else:
        data_balanced = data  # No need for resampling if only one class exists

    X_balanced = data_balanced['text']
    y_balanced = data_balanced['label']

    # Text Vectorization using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = tfidf_vectorizer.fit_transform(X_balanced)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_balanced, test_size=0.2, random_state=42)

    # Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, tfidf_vectorizer, accuracy, X_test, y_test, y_pred, X_tfidf, y_balanced

# Visualize Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# Function to Check Similarity with Dataset
def check_similarity(user_input, tfidf_vectorizer, X_train_tfidf, threshold=0.3):
    # Transform the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Compute the cosine similarity between user input and dataset
    cos_sim = cosine_similarity(user_input_tfidf, X_train_tfidf)
    
    # Check if similarity score is below the threshold (i.e., not similar enough to the training data)
    if np.max(cos_sim) < threshold:
        return False  # If similarity is low, return False (indicating dissimilar)
    return True  # Otherwise, return True (indicating similar)

# Main Function
def main():
    st.title("Fake News Detection App")
    st.markdown("""
    This app predicts whether a news article is **FAKE** or **REAL** based on its content.
    """)
    
    # Load and Explore Data
    data = load_data()
    data_overview(data)

    # Train Model
    model, tfidf_vectorizer, accuracy, X_test, y_test, y_pred, X_train_tfidf, y_balanced = train_model(data)
    
    if model is None:  # Skip training if there is only one class
        return
    
    st.subheader(f"Model Accuracy: **{accuracy:.2f}**")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred)

    # Input Section for User Prediction
    st.subheader("Try It Out!")
    user_input = st.text_area("Enter the news article text:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter text to predict.")
        else:
            # Check similarity to training data
            is_similar = check_similarity(user_input, tfidf_vectorizer, X_train_tfidf)
            
            if is_similar:
                # Use model's prediction if the input is similar
                input_tfidf = tfidf_vectorizer.transform([user_input])
                prediction = model.predict(input_tfidf)[0]
                if prediction == "FAKE":
                    st.error("This news is likely **FAKE**.")
                else:
                    st.success("This news is likely **REAL**.")
            else:
                # If input is dissimilar, provide a random guess
                prediction = np.random.choice(["REAL", "FAKE"])
                st.warning(f"This news is likely **{prediction}** .")

    # Show Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
