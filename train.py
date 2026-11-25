"""Train and evaluate a three-class tweet classifier."""
import html
import re
from pathlib import Path

import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).with_name("Tweets.csv")
LABELS = ["negative", "neutral", "positive"]
TOKENIZER = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def clean_tweet(text):
    """Remove links and handles, keeping words, negations and emoticons."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", html.unescape(text), flags=re.I)
    return " ".join(TOKENIZER.tokenize(text))


def load_tweets():
    tweets = pd.read_csv(DATA_FILE, usecols=["text", "airline_sentiment"]).dropna()
    tweets = tweets[tweets["airline_sentiment"].isin(LABELS)].copy()
    tweets["text"] = tweets["text"].map(clean_tweet)
    tweets = tweets[tweets["text"] != ""]
    # Remove ambiguous examples, then keep one copy of each cleaned tweet.
    label_count = tweets.groupby("text")["airline_sentiment"].transform("nunique")
    return tweets[label_count == 1].drop_duplicates("text")


def train_model():
    tweets = load_tweets()
    x_train, x_test, y_train, y_test = train_test_split(
        tweets["text"], tweets["airline_sentiment"],
        test_size=0.2, random_state=42, stratify=tweets["airline_sentiment"]
    )
    # The text has already been tokenized by NLTK. Split on spaces to reuse
    # those same tokens, including emoticons. Learn single words and word pairs.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                                  lowercase=False, ngram_range=(1, 2), min_df=2)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "correct": int((predictions == y_test).sum()),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "counts": tweets["airline_sentiment"].value_counts().reindex(LABELS),
        "report": classification_report(y_test, predictions, labels=LABELS, digits=3),
    }
    return model, results


if __name__ == "__main__":
    if not DATA_FILE.is_file():
        raise SystemExit(f"Missing {DATA_FILE.name}. See step 5 of README.md.")
    _, results = train_model()
    print(results["counts"].to_string())
    print(f"Training tweets: {results['train_size']}")
    print(f"Test accuracy: {results['correct']}/{results['test_size']} "
          f"({results['accuracy']:.2%})")
    print(results["report"])
