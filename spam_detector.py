import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=["label", "text"])
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data

def train_model(data):
    x_text = data["text"]
    y_label = data["label"]
    vectorizer = TfidfVectorizer()
    x_text = vectorizer.fit_transform(x_text)
    x_text_train, x_text_test, y_label_train, y_label_test = train_test_split(x_text, y_label, test_size=0.2)
    model = MultinomialNB()
    model.fit(x_text_train, y_label_train)
    y_pred = model.predict(x_text_test) 
    accuracy = accuracy_score(y_label_test, y_pred)
    report = classification_report(y_label_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    data = load_data()
    accuracy, report = train_model(data)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n {report}")

