# Twitter Sentiment Analysis

This internship project reads an English tweet and predicts **positive**, **negative**, or **neutral** sentiment. It was trained on historical tweets about US airlines. A traveller thanking the crew and a traveller complaining about a cancellation give the model different clues.

## Objectives and tools

- Clean labelled tweets with Pandas and NLTK.
- Train a TF-IDF and Logistic Regression classifier with scikit-learn, then measure its accuracy on unseen tweets.
- Use Streamlit to accept a new tweet and display a class distribution graph and test accuracy.

## Files and data

| File | Purpose |
| --- | --- |
| `Tweets.csv` | The 14,640 original rows, limited to tweet text and sentiment label. |
| `train.py` | Cleans, splits, trains and evaluates. Run this first to see the results. |
| `app.py` | Accepts a tweet, predicts sentiment and draws the chart. |
| `requirements.txt` | Lists the four Python packages to install. |
| `README.md` | Setup, code explanation, results and viva preparation. |

**Dataset credit.** CrowdFlower (Figure Eight), *Twitter US Airline Sentiment*. The bundled two-column CSV is an adaptation of the public dataset under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Only the column selection and CSV formatting were changed. Source and data card: [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) · [mirror with license metadata](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment).

## Run on Windows 11 with VS Code

1. Install 64-bit Python 3.12. The [Python 3.12.10 release page](https://www.python.org/downloads/release/python-31210/) has a **Windows installer (64-bit)** under Files. In the installer, select **Add python.exe to PATH** and enable the Python launcher if offered. Select **Install Now**.
2. Install [VS Code for Windows x64](https://code.visualstudio.com/download). Open it, press **Ctrl+Shift+X**, search for **Python**, and install the extension published by Microsoft.
3. In File Explorer, create an **Internship Projects** folder inside Documents. Right-click the project ZIP, select **Extract All**, and choose your new folder as the destination. In VS Code select **File > Open Folder** and open the extracted project folder that directly contains both Python files. If you are building by hand, create a folder with the repository name and copy the supplied files into it using **File > New Text File** and **Save As**.
4. Select **Terminal > New Terminal**. Use PowerShell. Run the commands below one line at a time. The first command should print Python 3.12.x. The environment keeps this project's packages together. Calling its Python directly also works when PowerShell script activation is disabled.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. The ZIP includes `Tweets.csv`. If you need a fresh copy, run the following command in the project terminal. It downloads the original CSV; the code reads the same two columns.

```powershell
curl.exe -fL "https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv" -o Tweets.csv
```

6. Train and test the model. Look for the test accuracy and the table for each class in the terminal.

```powershell
.\.venv\Scripts\python.exe train.py
```

7. Launch the app in the same terminal.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

8. Open the **Local URL** printed in the terminal (usually http://localhost:8501). Type your example in the box and click the prediction button. Scroll down to see the accuracy and the distribution graph. Leave the terminal open while using the app. Press **Ctrl+C** in the terminal to stop it. On later visits, repeat step 7.

The terminal training run and the app are two entry points to the same function. The app trains once when it starts, holds the model in memory, and reuses it between clicks. You can launch the app directly after setup. A saved model file is optional for this design.

**If a command fails**

- If Windows cannot find `py`, reopen VS Code after installing Python. If `python --version` already prints 3.12.x, use `python -m venv .venv` in step 4.
- If Python cannot find `requirements.txt` or `train.py`, run `dir` in the terminal. Open the folder that contains those files.
- If VS Code underlines an import after installation, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select the project's `.venv`.

## Follow the code

1. **Read and clean.** In `train.py`, `load_tweets()` reads the text and its human-assigned label. `clean_tweet()` decodes HTML and removes links. NLTK's `TweetTokenizer` removes @handles, lowercases words, shortens long repeated letters and splits the tweet into pieces. It keeps clues such as emoticons and “not”. It needs no separate NLTK downloads.
2. **Remove repeats.** If one cleaned tweet has conflicting labels, remove all its copies. If its labels agree, keep one copy. This prevents identical cleaned messages from appearing on both sides of the split.
3. **Make a fair test.** `train_test_split` keeps roughly 80% for training and 20% for testing. Stratification preserves the class proportions. The fixed seed, 42, repeats the split on the same data in the same order.
4. **Turn text into numbers.** TF-IDF assigns a weight to each known word or adjacent word pair. Words found across many different tweets receive less emphasis. The training tweets determine the vocabulary. A pair such as “not happy” gives the model information that “happy” alone misses.
5. **Learn the labels.** Logistic Regression learns one set of weights for each class. It combines the TF-IDF values with those weights and picks the class with the highest score. Balanced class weights give smaller classes more influence during training.
6. **Evaluate.** The pipeline predicts labels for the held-out tweets. `accuracy_score` compares predictions with the actual labels. The printed report also shows precision (how often a chosen label was right) and recall (how many actual examples of that class were found).
7. **Predict in the app.** `app.py` calls the same cleaner, then passes the cleaned tweet to the fitted pipeline. Blank input or input with no known features prompts the user to try again. The bar chart counts the true labels in the cleaned dataset.

## Measured result and demo

The supplied data yielded **14,236** usable tweets: 9,065 negative, 2,960 neutral and 2,211 positive. Training used **11,388**. The model correctly classified **2,266 of 2,848** held-out tweets (**79.56%**). Always guessing negative would score **63.69%** on this test set. Neutral was the hardest class here, with **66.4% recall**. These figures describe this split of airline tweets. Sarcasm and tweets about unrelated subjects can give different results.

| Try this tweet | Observed prediction |
| --- | --- |
| Thank you for the wonderful flight! | positive |
| My flight was cancelled again. Terrible service. | negative |
| What time does my flight depart? | neutral |

## 10 viva questions

1. **What does the project do?** It predicts whether an English tweet expresses positive, negative or neutral sentiment.
2. **What is the target column?** `airline_sentiment` is the answer the model learns to predict. The input column is `text`.
3. **Why use NLTK?** Its TweetTokenizer understands tweet features such as @mentions and emoticons.
4. **Why keep the word “not”?** “Happy” and “not happy” have different meanings. Keeping negation and word pairs lets the model learn that difference.
5. **What is TF-IDF?** It turns text into numerical weights based on how often a term occurs in a message and how widespread it is across the training collection.
6. **Why Logistic Regression?** It is a simple classifier that can learn weights for many text features and choose among three labels.
7. **Why split before fitting TF-IDF?** The test tweets must stay unseen. Otherwise their words would help decide the features used during training.
8. **Why use class weights?** Negative tweets are more common. The weights give mistakes on the smaller classes more influence during training.
9. **What does 79.56% mean?** It means 2,266 out of 2,848 held-out tweets received the same label as the dataset.
10. **Where can it fail?** A sarcastic tweet like “Wonderful, another delay” contains mixed clues. An airline-trained model may also struggle with sports slang.

## Publish on GitHub

1. Sign in to [GitHub](https://github.com/) or create a free account. Open [New repository](https://github.com/new).
2. Enter **twitter-sentiment-analysis**, choose **Public** so your reviewer can open it, leave the initialization options unchecked, and click **Create repository**.
3. On the empty repository page, click **uploading an existing file**. If the repository already has a file, choose **Add file > Upload files**.
4. Open the extracted folder in File Explorer. Select only **app.py, train.py, requirements.txt, Tweets.csv and README.md** and drag them into the browser together. This puts the README and source code at the repository root.
5. Enter a message such as **Add working classifier and README**, then click **Commit changes**. Check that GitHub displays the README below the file list. Copy the repository URL from your browser.
6. Repeat for the other internship project using its own repository name. Submit both repository links by email with your offer letter attached privately, as requested by Codec Technologies.

[GitHub's file upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
