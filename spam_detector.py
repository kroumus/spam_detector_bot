import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=["label", "text"])
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data

if __name__ == "__main__":
    data = load_data()
    print(data) 

def train_model(data):
    x_text = data["text"]
    y_label = data["label"]
    vectorizer = TfidfVectorizer()
    x_text = vectorizer.fit_transform(data_text)
    x_text_train, x_text_test, x_label_train, y_label_test = train_test_split(x_text, y_label, test_size=0.2)
    model = MultinomialNB()
    model.fit(x_text, y_label)
    y_pred = model.predict(x_text_test) 
    accuracy = accuracy_score