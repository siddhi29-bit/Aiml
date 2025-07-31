import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load datasets
users = pd.read_csv("users.csv")
posts = pd.read_csv("posts.csv")

# Load spam detection model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function: Post Recommendation
def recommend_posts(user_interest, posts):
    posts['content'] = posts['caption'] + ' ' + posts['hashtags']
    tfidf = TfidfVectorizer()
    post_vectors = tfidf.fit_transform(posts['content'])
    user_vector = tfidf.transform([user_interest])
    similarity = cosine_similarity(user_vector, post_vectors)
    top_indices = similarity.argsort()[0][-5:][::-1]
    return posts.iloc[top_indices]

# Streamlit UI
st.set_page_config(page_title="Instagram ML Project", layout="wide")
st.title("📱 Instagram Post Recommender & Spam Detector")

option = st.radio("Choose a feature:", ["📌 Post Recommendation", "🚫 Spam Detection"])

# Recommendation Feature
if option == "📌 Post Recommendation":
    st.subheader("🔍 Recommend Posts Based on User Interest")
    user_name = st.selectbox("Select User:", users['name'].unique())
    user_interest = users[users['name'] == user_name]['interests'].values[0]
    st.write(f"💡 User Interest: **{user_interest}**")
    
    recommended = recommend_posts(user_interest, posts)
    st.markdown("### ✅ Top 5 Recommended Posts:")
    for _, row in recommended.iterrows():
        st.write(f"**📷 Caption:** {row['caption']}")
        st.write(f"**🏷️ Hashtags:** {row['hashtags']}")
        st.write("---")

# Spam Detection Feature
elif option == "🚫 Spam Detection":
    st.subheader("🛡️ Detect Spam Comments")
    user_input = st.text_area("💬 Enter your comment below:")

    if st.button("Check Comment"):
        if user_input.strip() == "":
            st.warning("Please enter a comment.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            st.success(f"Prediction: **{prediction.upper()}**")
