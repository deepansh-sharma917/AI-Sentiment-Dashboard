import os
import re
import hashlib
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from streamlit_chat import message
from kaggle.api.kaggle_api_extended import KaggleApi

from sentiment_analysis import analyze_sentiment
from llm_chatbot import chatbot_response
from controllers.review_controller import insert_reviews_from_df, fetch_reviews
from database import get_db, Base, engine
from models import Review

# ---------------- DATABASE SETUP ----------------
Base.metadata.create_all(bind=engine)
db = next(get_db())

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🎭 AI Sentiment Analysis Dashboard", layout="wide")
st.title("🎭 AI Sentiment Analysis Dashboard")
st.write("Analyze IMDB reviews with AI-powered sentiment analysis, visualization, and chatbot insights.")

# ---------------- DATASET PATH ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CSV_FILE = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# ---------------- DOWNLOAD DATASET IF MISSING ----------------
if not os.path.exists(CSV_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    st.info("Downloading dataset from Kaggle... ⏳")
    api = KaggleApi()
    api.authenticate()  # Reads kaggle.json from default location
    api.dataset_download_files(
        'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
        path=DATA_DIR,
        unzip=True
    )
    st.success("✅ Dataset downloaded!")

# ---------------- LOAD CSV + SENTIMENT ----------------
@st.cache_data(show_spinner=False)
def load_csv(csv_path=CSV_FILE):
    df = pd.read_csv(csv_path)
    df['reviewText'] = df['review'].astype(str)

    processed_reviews, sentiment_list = [], []
    n = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, text in enumerate(df['reviewText'], start=1):
        clean_text = re.sub(r"<.*?>", " ", text)
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text.strip().lower())
        processed_reviews.append(clean_text)

        sentiment, score = analyze_sentiment(clean_text)
        sentiment_list.append((sentiment, score))

        if i % 500 == 0 or i == n:
            progress_bar.progress(i / n)
            status_text.text(f"🔍 Processing review {i}/{n}...")

    df['reviewText'] = processed_reviews
    df['review_hash'] = df['reviewText'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    df['sentiment'], df['sentiment_score'] = zip(*sentiment_list)

    progress_bar.empty()
    status_text.empty()
    return df

with st.spinner("Loading dataset... Please wait ⏳"):
    df = load_csv()
st.success("✅ Dataset loaded successfully!")

df = df.rename(columns={'reviewText': 'review_text'})

# ---------------- INSERT REVIEWS ----------------
existing_hashes = set(r[0] for r in db.query(Review.review_hash).all())
df_to_insert = df[~df['review_hash'].isin(existing_hashes)]
if not df_to_insert.empty:
    insert_reviews_from_df(df_to_insert, db)

# ---------------- FETCH REVIEWS ----------------
all_reviews = fetch_reviews(db)
reviews_df = pd.DataFrame([{
    "review_text": r.review_text,
    "sentiment": r.sentiment,
    "sentiment_score": r.sentiment_score
} for r in all_reviews])

# ---------------- KPI SECTION ----------------
st.subheader("📈 Key Performance Indicators (KPIs)")
total_reviews = len(reviews_df)
avg_score = reviews_df['sentiment_score'].mean()
sentiment_counts = reviews_df['sentiment'].value_counts()
pos_pct = (reviews_df['sentiment'] == 'POSITIVE').mean() * 100
neg_pct = (reviews_df['sentiment'] == 'NEGATIVE').mean() * 100
neu_pct = (reviews_df['sentiment'] == 'NEUTRAL').mean() * 100
sentiment_ratio = sentiment_counts.get('POSITIVE', 0) / max(sentiment_counts.get('NEGATIVE', 1), 1)
avg_length = reviews_df['review_text'].str.split().apply(len).mean()

col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Reviews", total_reviews)
col2.metric("💡 Avg Sentiment Score", f"{avg_score:.3f}")
col3.metric("🌟 Dominant Sentiment", sentiment_counts.idxmax())

col4, col5, col6 = st.columns(3)
col4.metric("😊 Positive Reviews (%)", f"{pos_pct:.2f}%")
col5.metric("😞 Negative Reviews (%)", f"{neg_pct:.2f}%")
col6.metric("⚖️ Sentiment Ratio (Pos/Neg)", f"{sentiment_ratio:.2f}")

st.metric("🕒 Avg Review Length (words)", f"{avg_length:.1f}")

# ---------------- SENTIMENT DISTRIBUTION ----------------
st.subheader("📊 Sentiment Overview")
st.bar_chart(reviews_df['sentiment'].value_counts())

st.subheader("🥧 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
ax.axis('equal')
st.pyplot(fig)
plt.close(fig)

# ---------------- FILTER + DATAFRAME ----------------
st.sidebar.header("🔍 Filter by Sentiment")
selected = st.sidebar.multiselect("Select Sentiments", reviews_df['sentiment'].unique(), reviews_df['sentiment'].unique())
filtered_df = reviews_df[reviews_df['sentiment'].isin(selected)]
st.dataframe(filtered_df[['review_text', 'sentiment', 'sentiment_score']].head(10))

# ---------------- TOP WORDS BY SENTIMENT ----------------
st.subheader("🧠 Top Words by Sentiment")
def get_top_words(df, sentiment_label, n=10):
    text_data = df[df['sentiment'] == sentiment_label]['review_text']
    if text_data.empty:
        return pd.DataFrame(columns=['word', 'count'])
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(text_data)
    word_counts = X.toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()
    freq_df = pd.DataFrame({'word': words, 'count': word_counts})
    return freq_df.sort_values(by='count', ascending=False).head(n)

sentiments = filtered_df['sentiment'].unique()
for sentiment_label in sentiments:
    top_words_df = get_top_words(filtered_df, sentiment_label)
    if not top_words_df.empty:
        st.write(f"### 🔹 Top Words in **{sentiment_label}** Reviews")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='count', y='word', data=top_words_df, hue="word", palette="crest", ax=ax, dodge=False, legend=False)
        ax.set_xlabel("Word Frequency")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close(fig)

# ---------------- ADVANCED INSIGHTS ----------------
st.markdown("## 📊 Advanced Sentiment Insights")

# --- Simulate time trend ---
if 'timestamp' not in reviews_df.columns:
    np.random.seed(42)
    start_date = datetime.date(2020, 1, 1)
    reviews_df['timestamp'] = [start_date + datetime.timedelta(days=int(i)) for i in np.random.randint(0, 365, size=len(reviews_df))]

st.subheader("📅 Sentiment Trend Over Time")
trend_df = reviews_df.groupby('timestamp')['sentiment_score'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trend_df['timestamp'], trend_df['sentiment_score'], linewidth=2)
ax.set_title("Average Sentiment Score Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Sentiment Score")
st.pyplot(fig)
plt.close(fig)

st.subheader("✍️ Sentiment vs. Review Length")
reviews_df['review_length'] = reviews_df['review_text'].str.split().apply(len)
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x='review_length', y='sentiment_score', hue='sentiment', data=reviews_df, ax=ax, alpha=0.6)
ax.set_title("Review Length vs. Sentiment Score")
ax.set_xlabel("Review Length (Words)")
ax.set_ylabel("Sentiment Score")
st.pyplot(fig)
plt.close(fig)

# ---------------- CHATBOT ----------------
st.subheader("💬 AI Chatbot — Ask about your reviews")
user_input = st.text_input("Type your question here:", key="chat_input")
if st.button("Send") and user_input:
    response = chatbot_response(user_input, filtered_df)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"user": user_input, "ai": response})

if "chat_history" in st.session_state:
    for chat in st.session_state.chat_history:
        message(chat["user"], is_user=True)
        message(chat["ai"], is_user=False)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, FastAPI, PostgreSQL & VADER Sentiment")
