### Assignment Overview.
This assignment describes how to develop a Python-based machine learning-based email spam classification system. To detect spam, the system uses a collection of emails to train a Naive Bayes model. Its features include email content preprocessing, model training, evaluation, and prediction.

## Stages of Assignment. 

Dataset Handling: The email spam dataset is available on kaggle.

Evaluation: The model's performance is evaluated using a confusion matrix, classification report, and F1 score.

Model Training: The MultinomialNB classifier is used to train the model on the processed email data.

Prediction: The trained model predicts if a new email is spam or not.

Preprocessing: Email text is tokenized and cleaned using NLTK's SpaceTokenizer.


## Required Libraries and Dataset.
- pandas
-  numpy
- scikit-learn
-  nltk
  The data set is used is the Email Spam Classification dataset from Keggle("Dataset":Contains 5172 emails with labels: 1 for spam, 0 for not spam.
Features are the frequency of 3000 words used in the emails.)

## Evaluation and Implimentation.
The dataset  is splitted into training and test sets, trains a Naive Bayes classifier, and evaluates its performance with accuracy and a detailed classification report as Evaluation Metrics:
-Confusion Matrix
-Classification Report (precision, recall, F1 score)
-F1 Score (balances precision and recall).

# Prediction of emails as spam or not spam.
# Preprocesses the email messages by tokenizing, removing stopwords, and applying stemming using NLTK

 
