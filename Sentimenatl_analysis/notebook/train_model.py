import pandas as pd
import joblib
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🔹 Download once (safe to keep)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# 🔹 NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# 🔥 Load CSV WITHOUT header
df = pd.read_csv(
    "data/sentiment.csv",
    header=None,
    names=["id", "entity", "sentiment", "text"]
)

# Keep only text + sentiment
df = df[["text", "sentiment"]].dropna()

# Keep valid classes
df = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]

# 🔥 APPLY NLP PREPROCESSING
df["clean_text"] = df["text"].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# ML pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model/sentiment_model.pkl")

print("✅ Multi-class sentiment model trained with NLP preprocessing")
