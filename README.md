# machine-learning-algorithms-implemented-for-SMS-spam-message-detection-on-UCI-spam-message-dataset

# SMS Spam Detection

A machine learning project that classifies SMS messages as **spam** or **ham** using multiple algorithms. Built to demonstrate text preprocessing, feature extraction, and model evaluation.

## Project Overview
The goal of this project is to detect spam messages from the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). The project implements **Logistic Regression, Naive Bayes, and Decision Tree** algorithms to compare performance and accuracy.

## Features
- **Text Preprocessing**: Lowercasing, removing punctuation and numbers.  
- **Feature Extraction**: TF-IDF vectorization for numerical representation of text.  
- **Machine Learning Models**:  
  - Logistic Regression  
  - Naive Bayes  
  - Decision Tree  
- **Model Evaluation**: Accuracy score and detailed classification report.  

## Technologies Used
- Python  
- pandas, NumPy  
- scikit-learn  
- Regular expressions (re)  

## Dataset
- UCI SMS Spam Collection dataset (tab-separated, labels: `ham` or `spam`)  
- Labels converted to binary (`0` = ham, `1` = spam)

## How to Run
1. code was run in google colab research lab 
