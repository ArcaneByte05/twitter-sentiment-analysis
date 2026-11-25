"""Run with: python -m streamlit run app.py"""
import streamlit as st

from train import DATA_FILE, clean_tweet, train_model

st.title("Twitter sentiment analysis")
st.write("How does this tweet sound? The model learned from English airline tweets.")

if not DATA_FILE.exists():
    st.error("Put Tweets.csv beside train.py, then refresh. See README.md for the download.")
    st.stop()

# Train once at startup and reuse the model when the user clicks a button.
@st.cache_resource
def get_model():
    return train_model()


with st.spinner("Training on the labelled tweets..."):
    model, results = get_model()

tweet = st.text_area("Enter a tweet", placeholder="Thanks for getting me home on time!")
if st.button("Predict sentiment"):
    text = clean_tweet(tweet)
    if not text:
        st.warning("Please enter a tweet with some text.")
    elif model.named_steps["tfidf"].transform([text]).nnz == 0:
        st.warning("The model has never seen these words. Try a longer English tweet.")
    else:
        label = model.predict([text])[0]
        st.success(f"Predicted sentiment: {label}")

st.subheader("Test results")
st.metric("Held-out accuracy", f"{results['accuracy']:.2%}")
st.caption(f"{results['correct']:,} correct out of {results['test_size']:,} test tweets. "
           f"Trained on {results['train_size']:,} other tweets.")
st.subheader("Sentiment distribution")
st.caption("Number of tweets in each class after cleaning, before the split.")
st.bar_chart(results["counts"].rename("Tweets"))
