# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# Dataset is loaded
file_path = '/content/smsspamcollection.zip'
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
# Assuming columns 'message' and 'label'




# Text vectorization is done using TF-IDF
vectorize = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorize.fit_transform(df['message'])




# Labels (0 = ham, 1 = spam)
y = df['label']




# Data is split into training and test sets
Xtrainset, Xtestset, ytrainset, ytestset = train_test_split(X, y, test_size=0.2, random_state=42)




# Naive Bayes classifier is initialized
nb_classifier = MultinomialNB()




# Classifier is trained
nb_classifier.fit(Xtrainset, ytrainset)




# Predictions are made on the test set
y_predictions = nb_classifier.predict(Xtestset)




# Evaluating the model
print("Accuracy:", accuracy_score(ytestset, y_predictions))
print("Confusion Matrix:\n", confusion_matrix(ytestset, y_predictions))
print("Classification Report:\n", classification_report(ytestset, y_predictions))




