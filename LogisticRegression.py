import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
file_path = "/content/smsspamcollection.zip"


#dataset loaded with tab separator
df = pd.read_csv(file_path, header=None, sep='\t', names=['label', 'message'])


# labels are converted to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Data cleaning function
def text_preprocess(txt):
    txt = txt.lower()  # lowercases
    txt = re.sub(r'\d+', '', txt)  # numbers are removed
    txt = txt.translate(str.maketrans('', '', string.punctuation))  # punctuation is removed
    return txt


df['message'] = df['message'].apply(text_preprocess)


# dataset is splitted
X_trainset, X_testset, y_trainset, y_testset = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)




# text is converted to numerical features using TF-IDF
vectorize = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorize.fit_transform(X_trainset)
X_test_tfidf = vectorize.transform(X_testset)


# Logisitic regression model is trained
model = LogisticRegression()
model.fit(X_train_tfidf, y_trainset)


# predictions are made
y_predictions = model.predict(X_test_tfidf)


# model is evaluated
acc = accuracy_score(y_testset, y_predictions)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_testset, y_predictions))
