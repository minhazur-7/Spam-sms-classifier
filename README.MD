# 📧 Spam SMS Classifier

A machine learning project that classifies text messages as **Spam** or **Ham** (not spam), built with Python and scikit-learn. This is my final project for the Classification module, where I practiced the full workflow: cleaning data, converting text to numbers, comparing six algorithms, and analyzing the model's mistakes.

---

## 🎯 Goal

Build a spam filter that **rarely marks real messages as spam**, because losing an important message (an OTP, a job call) is worse than seeing an extra spam message.

That is why this project focuses on **Precision** and **Recall** instead of accuracy alone.

---

## 📂 Dataset

- **Source:** [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages, reduced to **5,169** after removing duplicates
- **Labels:** `ham` (0) and `spam` (1)
- **Challenge:** the data is **imbalanced**. A model that predicts "ham" for everything would still score about 87% accuracy, so accuracy alone is misleading.

---

## 🔄 Workflow

1. **Load and clean:** drop unused columns, rename columns, remove duplicates
2. **Encode labels:** ham → 0, spam → 1
3. **Train-test split:** 80/20, stratified, split *before* vectorizing to avoid data leakage
4. **Text to numbers:** `CountVectorizer` (Bag of Words), fit on training data only (7,591 unique words)
5. **Train six models:** Naive Bayes, Logistic Regression, SVM, Random Forest, Decision Tree, KNN
6. **Evaluate:** Confusion Matrix, Precision, Recall, F1-score, ROC-AUC
7. **Error analysis:** read the messages the model got wrong
8. **Predict:** a `predict_spam(text)` function for new messages

---

## 📊 Results (spam class, test set of 1,034 messages)

| Model | Precision | Recall | F1 | FP | FN | ROC-AUC |
|---|---|---|---|---|---|---|
| SVM (linear) | 1.000 | 0.878 | 0.935 | 0 | 16 | 0.976 |
| KNN | 1.000 | 0.275 | 0.431 | 0 | 95 | 0.810 |
| Random Forest | 0.990 | 0.794 | 0.881 | 1 | 27 | 0.992 |
| Logistic Regression | 0.991 | 0.847 | 0.914 | 1 | 20 | 0.979 |
| **Naive Bayes** | **0.976** | **0.916** | **0.945** | **3** | **11** | **0.982** |
| Decision Tree | 0.906 | 0.878 | 0.891 | 12 | 16 | 0.932 |

**Final model: Multinomial Naive Bayes.** It has the best F1 and Recall, while keeping Precision high (only 3 real messages wrongly marked as spam). SVM is a strong alternative if zero false positives is the top priority.

---

## 💡 Key Learnings

- **Accuracy can lie on imbalanced data.** KNN has 0 false positives but catches only 27% of spam. Low FP alone does not make a good model.
- **Precision vs Recall is a trade-off,** and the right balance depends on the real-world cost of each mistake.
- **Fit the vectorizer on training data only.** Fitting on the test set leaks information.
- **Bag of Words ignores context.** The word "call" caused one mistake in each direction:
  - *"Waiting for your call."* (ham) was wrongly flagged as spam
  - *"Missed call alert..."* (spam) was missed
- **Missed spam often looks like a normal chat message** or contains phone numbers and short codes that the model cannot generalize from.

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn · seaborn · matplotlib

---

## ▶️ How to Run

```bash
# 1. Install libraries
pip install pandas scikit-learn seaborn matplotlib

# 2. Download spam.csv from Kaggle and place it here:
#    classification/spam_email_classifier/spam.csv

# 3. Run
python classification/spam_email_classifier/train.py
```

### Example

```python
predict_spam("Congratulations! You won a free prize, call now to claim")
# ('Spam', 0.99)

predict_spam("Bro, are we meeting at 5 today?")
# ('Ham', 0.01)
```

---