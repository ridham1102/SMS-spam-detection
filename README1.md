# 📱 SMS Spam Detection with Word2Vec + Random Forest

This project implements an SMS spam detection system using **Word2Vec embeddings** and a **Random Forest Classifier**. Unlike traditional approaches that work on message-level data, this solution breaks each SMS into sentences, processes them using Word2Vec, and then applies a machine learning model to classify whether each sentence is spam or not.

---

## 🧠 Project Objective

The goal is to build a robust classifier that can automatically detect spam SMS content by learning from historical labeled SMS data. The model operates at the **sentence level**, improving granularity and potentially allowing partial message spam detection.

---

## 📂 Dataset

The dataset used is the **[SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)**. It contains over 5,000 SMS messages labeled as either:

- `ham` (not spam)
- `spam` (unsolicited or malicious content)

---

## 🛠️ Tech Stack

- **Python**
- **NLTK** for tokenization and lemmatization
- **Gensim** for Word2Vec embedding
- **Scikit-learn** for Random Forest classification
- **Pandas / NumPy** for data manipulation
- **TQDM** for progress bars

---

## 🧾 Workflow

### 1. **Data Cleaning and Preprocessing**
- Loaded SMS messages and cleaned them by removing all non-alphabet characters.
- Split each message into individual **sentences** using `sent_tokenize`.
- Only retained sentences with English letters (A–Z, a–z).

### 2. **Text Vectorization using Word2Vec**
- Used `gensim.models.Word2Vec` to create dense word embeddings.
- Transformed each sentence into a **single fixed-length vector** by averaging the word vectors.
- Handled missing vectors gracefully by falling back to a zero-vector.

### 3. **Feature Matrix Creation**
- Resulting in a matrix of shape `(5769, 100)` — each row representing one sentence's embedding.

### 4. **Label Encoding**
- Labels (spam/ham) were aligned with each sentence (not full message).
- Encoded using one-hot approach → converted to binary values.

### 5. **Model Training**
- Trained a **Random Forest Classifier** on the Word2Vec vectors.
- Split data into training and test sets for performance evaluation.

### 6. **Model Evaluation**
- Evaluated using accuracy, precision, recall, and F1-score.
- Observed strong performance even on sentence-level classification.

---

## 📈 Sample Results

*(Add your actual results below if you have them)*

```text
Accuracy: 0.96
Precision: 0.94
Recall: 0.93
F1-Score: 0.935
