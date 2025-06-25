# 🩺 Symptom Tracker Web App

A Flask-based web application that takes symptom descriptions from users and predicts possible medical conditions using a machine learning model. The model is trained with TF-IDF weighted Word2Vec embeddings and a classifier to understand and categorize natural language descriptions of symptoms.

---

## Features

- Accepts natural language symptom descriptions (e.g., "I have a sore throat and fever").
- Preprocesses text using stemming, stopword removal, and tokenization.
- Uses **TF-IDF weighted Word2Vec** to convert input into semantic vectors.
- Predicts a disease label using a pre-trained ML classifier.
- Simple and intuitive HTML front-end with two pages:
  - `firstpage.html`: symptom input page
  - `resultpage.html`: displays prediction

---

##  Project Structure
symptom-tracker/
│
├── static/
│ └── style.css # CSS styles 
│
├── templates/
│ ├── firstpage.html # Input form for user symptoms
│ └── resultpage.html # Displays the predicted disease
│
├── model1.pkl # Trained classifier 
├── tfidf.pkl # Trained TF-IDF vectorizer
├── w2v_model.model # Trained Word2Vec model
├── app.py # Main Flask application
└── README.md # This file


---

## ⚙️ How It Works

1. **User Input**: User enters a symptom description in plain English.
2. **Preprocessing**:
   - Remove non-alphanumeric characters
   - Lowercasing, stemming, stopword removal
   - Tokenization using Gensim's `simple_preprocess`
3. **Vectorization**: Uses TF-IDF-weighted Word2Vec to convert text into a feature vector.
4. **Prediction**: The pre-trained model (`model1.pkl`) predicts a condition.
5. **Output**: The result is shown on a new page (`resultpage.html`).

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
dependencies:
  - python=3.9  # or your specific Python version
  - pip
  - pip:
      - Flask
      - numpy
      - scikit-learn
      - gensim
      - nltk

post_install:
  - python -c "import nltk; nltk.download('stopwords')"

run:
  - python app.py ```




