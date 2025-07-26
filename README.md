# 📱 SMS Spam Detection

This project demonstrates a basic SMS spam detection model using Python and scikit-learn. The goal is to classify SMS messages as either **ham** (not spam) or **spam** using traditional machine learning and NLP techniques.

---

## 📊 Overview

Every day, countless SMS messages are sent, and many are unsolicited spam. This notebook walks through building a simple spam filter that can automatically classify messages using a supervised learning approach.

The model is trained on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which contains 5,000+ labeled messages.

---

## 🔍 Methodology

The pipeline follows a structured process:

1. **Data Loading**  
   Load the dataset into a pandas DataFrame.

2. **Text Preprocessing**  
   Clean and prepare the text for model input:
   - Convert to lowercase  
   - Remove special characters and numbers  
   - Tokenize the text  
   - Remove stopwords  
   - Lemmatize each token

3. **Feature Extraction**  
   Convert text to numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

4. **Data Splitting**  
   Split the dataset into training and testing sets (commonly 80/20 or 70/30).

5. **Model Training**  
   Train a **Multinomial Naive Bayes** classifier — ideal for text classification.

6. **Model Evaluation**  
   Evaluate performance using accuracy score and a classification report (precision, recall, F1-score).

---

## 🛠️ Libraries Used

- `pandas` – Data manipulation  
- `numpy` – Numerical operations  
- `nltk` – NLP tasks (tokenization, stopword removal, lemmatization)  
- `re` – Regular expressions for text cleaning  
- `scikit-learn` – ML models, TF-IDF, train/test split, evaluation  
- `matplotlib` & `seaborn` *(optional)* – For data visualization (minimal use)

---

## 📁 File Structure

SMS-Spam-Detection/
│
├── SMS_Spam_Detection.ipynb # Jupyter Notebook with all steps
├── SMSSpamCollection.txt # Dataset file (place in root or /content/)
├── README.md # This documentation

---

## 🚀 How to Run

1. Make sure Python 3.x is installed.
2. Clone this repository or download the notebook.
3. Ensure the dataset file `SMSSpamCollection.txt` is in the same directory or `/content/` if using Colab.
4. Open the `SMS_Spam_Detection.ipynb` notebook.
5. Run each cell in sequence to see the model train and predict.

---

## 📈 Results

The model generally achieves over **95% accuracy** on test data. A classification report helps in analyzing precision and recall for both "spam" and "ham" classes.

---


