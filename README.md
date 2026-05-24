# Movie Review Sentiment Analyzer

A simple NLP-based web application that predicts whether a movie review is Positive or Negative using Machine Learning and Natural Language Processing techniques.

# Features
- Text preprocessing
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF Vectorization
- Naive Bayes Classification
- Interactive Streamlit Web App

# Technologies Used
- Python
- Streamlit
- NLTK
- Scikit-learn

# How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app

```bash
streamlit run app.py
```

# Project Workflow
1. User enters a movie review
2. Text preprocessing is applied
3. TF-IDF converts text into vectors
4. Naive Bayes model predicts sentiment
5. Output shows Positive or Negative review

# Output
- Positive Review 😊
- Negative Review 😞
