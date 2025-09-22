# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 2: dataset is loaded
file_path = '/content/smsspamcollection.zip'
# dataset is read into dataframe
df = pd.read_csv(file_path, header=None, sep='\t',  names=['label', 'message'])


# Step 3: data is preprocessed (labels are converted to binary: spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})


# Step 4: text data is vertorized (messages are converted to numerical data using TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])  # Features: the messages
y = df['label']  # Target: spam/ham labels


# Step 5: dataset is split into training and testing sets
Xtrainset, Xtestset, ytrainset, ytestset = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 6: decision tree classifier is intialized and trained
model = DecisionTreeClassifier(random_state=42)
model.fit(Xtrainset, ytrainset)  # Fit the model with the training data


# Step 7: predictions are made on test set
y_predictions = model.predict(Xtestset)


# Step 8: Evaluate the model's performance
acc = accuracy_score(ytestset, y_predictions)
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(ytestset, y_predictions))
print("Confusion Matrix:\n", confusion_matrix(ytestset, y_predictions))


# decision tree is visualized
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=tfidf.get_feature_names_out(), class_names=['ham', 'spam'])
plt.show()
