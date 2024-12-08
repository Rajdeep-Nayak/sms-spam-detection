# Overview
This project involves building a classification model to detect whether an SMS message is spam or not. The model uses a dataset of labeled SMS messages to train a machine learning model to classify new messages based on their content. The dataset consists of two classes: spam and non-spam messages. The goal is to create an efficient and accurate model capable of distinguishing between the two classes.

# Features
Preprocessing: The dataset is cleaned and preprocessed to remove unnecessary characters and ensure that the data is in a format suitable for training.
Text Vectorization: The text data is converted into numerical representations using methods like TF-IDF to enable model training.
Model Selection: We experimented with various models such as SVM, Logistic Regression, and Naive Bayes to determine the best-performing classifier for SMS spam detection.
Evaluation: We evaluated model performance using accuracy, precision, recall, F1-score, and confusion matrix.


# Dataset: 
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
The dataset used in this project is the SMS Spam Collection dataset, which contains a collection of SMS messages in both English and non-English languages. The dataset is publicly available and can be downloaded from Kaggle's SMS Spam Collection.

The dataset consists of 5574 SMS messages, of which 747 are spam messages, and 4827 are non-spam messages.

# Steps:

## Data Preprocessing
The raw SMS text data undergoes several preprocessing steps:

Lowercasing: All text is converted to lowercase to standardize the input.
Removing special characters and digits: We remove punctuation, numbers, and non-alphabetic characters.
Stopword Removal: Common English stopwords (such as "is", "and", "the") are removed.
Text Tokenization: The text is split into individual words (tokens).
Stemming: Words are reduced to their root form using a stemming algorithm.
Vectorization: The text data is converted into numerical format using the TF-IDF vectorizer.
Models
Several machine learning models were trained and evaluated for SMS spam detection:

Support Vector Machine (SVM)
The Support Vector Machine (SVM) classifier is used to separate the two classes (spam and non-spam) by finding the hyperplane that maximizes the margin between them.

Logistic Regression
A simple yet effective linear model was trained for classification.

Naive Bayes
A probabilistic classifier that applies Bayes' Theorem to predict the probability of a message being spam.

### Evaluation Metrics
After training the models, we evaluated them based on the following metrics:

Accuracy: The percentage of correct predictions.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positive predictions among all actual positives.
F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
Results
The following are the performance metrics of the Support Vector Machine (SVM) model:

### 1. Overall Accuracy
The model achieved an accuracy of 98.03% on the validation dataset.
### 2. SVM Classification Report
Non-Spam (0):

Precision: 0.98

Recall: 1.00

F1-Score: 0.99

Support: 965

Spam (1):


Precision: 1.00

Recall: 0.85

F1-Score: 0.92

Support: 150

3. Macro Average

Precision: 0.99

Recall: 0.93

F1-Score: 0.95

4. Weighted Average

Precision: 0.98

Recall: 0.98

F1-Score: 0.98

## Key Observations:
The model performs exceptionally well in identifying non-spam messages, with a recall of 1.00 for non-spam and a high precision of 0.98.
The recall for spam messages is somewhat lower at 0.85, meaning some spam messages are being missed, but precision is still high at 1.00, indicating that any identified spam messages are likely to be true positives.
Overall, the model demonstrates a strong balance between precision and recall, achieving good performance on both spam and non-spam categories.

## Conclusion
This project demonstrates the effectiveness of using machine learning models, specifically Support Vector Machine (SVM), for SMS spam detection. While the model achieves high accuracy and performs well on the non-spam category, there is room for improvement in detecting spam messages, particularly in reducing false negatives (missed spam). Future work could include tuning hyperparameters, using deep learning models, or exploring more advanced text preprocessing techniques to improve performance.
