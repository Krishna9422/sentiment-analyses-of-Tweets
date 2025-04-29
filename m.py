import tweepy
from flask import Flask, render_template, request
from transformers import pipeline
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tweepy.errors

app = Flask(__name__)

# Twitter API v2 setup
BEARER_TOKEN = ""
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

def fetch_and_classify_tweets(keyword, count=20):
    try:
        response = client.search_recent_tweets(
            query=keyword,
            max_results=100,
            tweet_fields=["text", "created_at"]
        )
        tweets = response.data or []
    except tweepy.TooManyRequests:
        return [], [], None, "Rate limit exceeded. Please wait and try again."
    except Exception as e:
        return [], [], None, f"Error fetching tweets: {str(e)}"

    positive_tweets = []
    negative_tweets = []
    pos_scores, neg_scores = [], []

    for tweet in tweets[:count]:
        text = tweet.text
        result = sentiment_pipeline(text[:512])[0]
        entry = {
            "text": text,
            "sentiment": result["label"],
            "score": result["score"],
            "date": tweet.created_at.strftime("%Y-%m-%d %H:%M") if tweet.created_at else "N/A"
        }

        if result["label"] == "POSITIVE":
            positive_tweets.append(entry)
            pos_scores.append(result["score"])
        elif result["label"] == "NEGATIVE":
            negative_tweets.append(entry)
            neg_scores.append(result["score"])

    positive_tweets.sort(key=lambda x: x["score"], reverse=True)
    negative_tweets.sort(key=lambda x: x["score"], reverse=True)

    # Create the sentiment confidence histogram
    plt.figure(figsize=(8, 5))
    if pos_scores:
        plt.hist(pos_scores, bins=10, alpha=0.6, color='green', label='Positive')
    if neg_scores:
        plt.hist(neg_scores, bins=10, alpha=0.6, color='red', label='Negative')

    plt.xlabel('Confidence Score')
    plt.ylabel('Tweet Count')
    plt.title(f"Sentiment Confidence Distribution for '{keyword}'")
    plt.legend()

    # Save plot in static folder
    plot_filename = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plot_path = os.path.join("static", plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return positive_tweets[:10], negative_tweets[:10], plot_path, None

@app.route("/", methods=["GET", "POST"])
def dashboard():
    keyword = "Digital India"
    if request.method == "POST":
        keyword = request.form.get("keyword", "Digital India")

    positive, negative, plot_path, error = fetch_and_classify_tweets(keyword)

    return render_template(
        "index.html",
        keyword=keyword,
        positive=positive,
        negative=negative,
        plot_path=plot_path,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
