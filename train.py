from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             precision_score,
                             recall_score,
                             f1_score)

df = pd.read_csv(
    "spam_email_classifier/spam.csv", encoding="latin-1")

# keep only necessary columns
df = df[['v1', 'v2']]

# rename column
df = df.rename(columns={
    'v1': 'label',
    'v2': "message"
})

# remove duplicates
df = df.drop_duplicates()

# create message length
df["message_length"] = df["message"].str.len()


# label encoding
df["label"] = df['label'].map(
    {
        'ham': 0,
        'spam': 1
    }
)

# x/y

x = df["message"]
y = df["label"]

# train test split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


vectorizer = CountVectorizer()

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    model.fit(x_train_vec, y_train)
    y_pred = model.predict(x_test_vec)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test_vec)[:, 1]

    else:
        y_prob = model.decision_function(x_test_vec)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "Model": name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TP": tp,
        "AUC": roc_auc_score(y_test, y_prob)
    })


results_df = pd.DataFrame(results).sort_values("FP")

print(results_df)
#example
'''final_model = MultinomialNB()
final_model.fit(x_train_vec, y_train)


def predict_spam(text):
    text_vec = vectorizer.transform([text])
    pred = final_model.predict(text_vec)[0]
    spam_prob = final_model.predict_proba(text_vec)[0][1]
    labell = "Spam" if pred == 1 else "Ham"
    return labell, spam_prob


text_messages = [
    "Congratulations! You won a free prize, call now to claim",
    "URGENT! Text WIN to 80082 to get a 500 pound reward",
    "Free entry in a weekly contest, reply YES to join",
    # normal
    "Bro, are we meeting at 5 today?",
    "Please send me the class notes when you are free",
    "Happy birthday! Let's have dinner this weekend",
    # tricky
    "Waiting for your call.",
    "Nokia phone is lovly..",
]

for msg in text_messages:
    labell , prob = predict_spam(msg)
    print(f"{labell:5} (spam prob: {prob:.2f}) -> {msg}")'''

