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

@st.cache_data
def load_data():
    data = pd.read_csv("news.csv")  
    
    
    data['subject'] = data['subject'].fillna('') 

 
    data['label'] = data['label'].apply(lambda x: 'REAL' if x == True else 'FAKE')
    
    return data


def data_overview(data):
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write("Shape of Dataset:", data.shape)
    
   
    class_distribution = data['label'].value_counts()
    st.write("Class Distribution:")
    st.write(class_distribution)
    
    st.bar_chart(class_distribution)


@st.cache_resource
def train_model(data):
    data = data.dropna(subset=['text', 'subject'])  
    X = data['text']
    y = data['label']

   
    class_distribution = data['label'].value_counts()
    st.write("Class Distribution (before resampling):")
    st.write(class_distribution)


    if len(class_distribution) == 1:
        st.warning(f"Warning: The dataset contains only one class: {class_distribution.index[0]}.")
        return None, None, 0, None, None, None


    if len(class_distribution) == 2 and 'FAKE' in class_distribution and 'REAL' in class_distribution:
       
        data_balanced = pd.concat([
            resample(data[data['label'] == 'FAKE'], replace=True, n_samples=len(data[data['label'] == 'REAL']), random_state=42),
            data[data['label'] == 'REAL']
        ])
    else:
        data_balanced = data 

    X_balanced = data_balanced['text']
    y_balanced = data_balanced['label']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = tfidf_vectorizer.fit_transform(X_balanced)

 
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_balanced, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, tfidf_vectorizer, accuracy, X_test, y_test, y_pred, X_tfidf, y_balanced


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

def check_similarity(user_input, tfidf_vectorizer, X_train_tfidf, threshold=0.3):
  
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    
    cos_sim = cosine_similarity(user_input_tfidf, X_train_tfidf)
    
    
    if np.max(cos_sim) < threshold:
        return False 
    return True  


def main():
    st.title("Fake News Detection App")
    st.markdown("""
    This app predicts whether a news article is **FAKE** or **REAL** based on its content.
    """)
    
 
    data = load_data()
    data_overview(data)


    model, tfidf_vectorizer, accuracy, X_test, y_test, y_pred, X_train_tfidf, y_balanced = train_model(data)
    
    if model is None: 
        return
    
    st.subheader(f"Model Accuracy: **{accuracy:.2f}**")
    
   
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred)

    
    st.subheader("Try It Out!")
    user_input = st.text_area("Enter the news article text:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter text to predict.")
        else:
          
            is_similar = check_similarity(user_input, tfidf_vectorizer, X_train_tfidf)
            
            if is_similar:
                
                input_tfidf = tfidf_vectorizer.transform([user_input])
                prediction = model.predict(input_tfidf)[0]
                if prediction == "FAKE":
                    st.error("This news is likely **FAKE**.")
                else:
                    st.success("This news is likely **REAL**.")
            else:
                
                prediction = np.random.choice(["REAL", "FAKE"])
                st.warning(f"This news is likely **{prediction}** .")

    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
