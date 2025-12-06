from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_data():
    data = {
        "text":[
            "The doctor was kind and helpful",
            "Terrible service and rude staff",
            "Amazing care, very satisfied",
            "The wait time was extremely long",
            "Great support from the nurse team",
            "Worst hospital experience ever"
        ],
        "label":["positive","negative","positive","negative","positive","negative"]
    }
    return pd.DataFrame(data)

def main():
    df = load_data()
    X = df["text"]
    y = df["label"]
    vec = CountVectorizer()
    Xv = vec.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xv,y,test_size=0.3,random_state=42)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,preds))
    print(classification_report(y_test,preds))

if __name__ == "__main__":
    main()
