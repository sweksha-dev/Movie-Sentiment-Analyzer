# 🎬 Movie Review Sentiment Analyzer (Web App Version)

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 🔹 Download NLTK data
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# 🔹 Preprocessing setup
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Tokenize, remove stopwords, lemmatize"""
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

# -----------------------------
# 🔹 Sample training data
# -----------------------------
texts = [
    "I love this product, it’s amazing!",
    "This is the worst movie ever.",
    "I really enjoyed this concert.",
    "I hate waiting in long lines.",
    "The service was fantastic!",
    "The food was terrible."
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Preprocess training text
clean_texts = [preprocess(t) for t in texts]

# TF-IDF + Naive Bayes model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_texts)
model = MultinomialNB()
model.fit(X, labels)

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎥")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("This app uses NLP steps you learned — Tokenization, Stopwords, Lemmatization, TF-IDF, and Naive Bayes!")

user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Show preprocessing steps
        tokens = word_tokenize(user_input.lower())
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

        # st.write("### 🧹 Step 1 – Tokenization:")
        # st.write(tokens)
        # st.write("### 🚫 Step 2 – After Stopword Removal:")
        # st.write(filtered)
        # st.write("### 🧠 Step 3 – After Lemmatization:")
        # st.write(lemmatized)

        # Predict sentiment
        processed = " ".join(lemmatized)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = round(max(proba) * 100, 2)

        st.markdown("---")
        st.subheader(f"🔍 Predicted Sentiment: {prediction.capitalize()}")
        st.write(f"**Confidence:** {confidence}%")

        if prediction == "positive":
            st.success("😊 Positive Review!")
        else:
            st.error("😞 Negative Review.")
