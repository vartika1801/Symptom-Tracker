# from flask import Flask, request, redirect, url_for, render_template

# app=Flask(__name__)

# import pickle
# with open('vectorizer.pkl','rb') as f:
#     tv=pickle.load(f)
# with open('model.pkl','rb') as f:
#     clf=pickle.load(f)

# @app.route("/",methods=['GET','POST'])
# def home():
#     prediction=None
#     if request.method == 'POST':
#         user_input=request.form['symptoms']
#         if user_input.strip() == "":
            
#             prediction="Please enter a valid description."
#         else:
#             transformed_input=tv.transform([user_input])
#             prediction = clf.predict(transformed_input)[0]
#         return render_template('resultpage.html',prediction=prediction)
#     return render_template('firstpage.html')


# if __name__=="__main__":
#     app.run(debug=True)
    
    
from flask import Flask, request, render_template
import pickle
import numpy as np
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Flask app setup
app = Flask(__name__)

# Load models
clf = pickle.load(open("model1.pkl", "rb"))             # Your trained classifier
tfidf = pickle.load(open("tfidf.pkl", "rb"))            # Trained TF-IDF vectorizer
w2v_model = Word2Vec.load("w2v_model.model")            # Trained Word2Vec model
tfidf_vocab = tfidf.vocabulary_                         # TF-IDF vocabulary

# Preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text Preprocessing
def preprocess(text):
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return simple_preprocess(" ".join(review))

# TF-IDF Weighted Word2Vec
def tfidf_weighted_w2v(tokens, tfidf_model, tfidf_vocab, w2v_model):
    vecs = []
    for word in tokens:
        if word in w2v_model.wv and word in tfidf_vocab:
            weight = tfidf_model.idf_[tfidf_vocab[word]]
            vecs.append(w2v_model.wv[word] * weight)
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_input = request.form["symptoms"]
        if user_input.strip() == "":
            prediction = "Please enter a valid description."
        else:
            tokens = preprocess(user_input)
            vector = tfidf_weighted_w2v(tokens, tfidf, tfidf_vocab, w2v_model).reshape(1, -1)
            prediction = clf.predict(vector)[0]
        return render_template("resultpage.html", prediction=prediction)
    return render_template("firstpage.html")

if __name__ == "__main__":
    app.run(debug=True)
