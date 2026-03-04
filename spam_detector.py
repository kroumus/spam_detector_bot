import pandas as pd

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=["label", "text"])
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data

if __name__ == "__main__":
    data = load_data()
    print(data) 