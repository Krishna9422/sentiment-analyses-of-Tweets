import tweepy
from flask import Flask, render_template, request
from transformers import pipeline
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

def fetch_and_classify_tweets(bearer_token, keyword, count=20):
    try:
        client = tweepy.Client(bearer_token=bearer_token)
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

    positive_tweets, negative_tweets = [], []
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

    # Create histogram
    plt.figure(figsize=(8, 5))
    if pos_scores:
        plt.hist(pos_scores, bins=10, alpha=0.6, color='green', label='Positive')
    if neg_scores:
        plt.hist(neg_scores, bins=10, alpha=0.6, color='red', label='Negative')
    plt.xlabel('Confidence Score')
    plt.ylabel('Tweet Count')
    plt.title(f"Sentiment Confidence Distribution for '{keyword}'")
    plt.legend()

    if not os.path.exists("static"):
        os.makedirs("static")
    plot_filename = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plot_path = os.path.join("static", plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return positive_tweets[:5], negative_tweets[:5], plot_path, None

@app.route("/", methods=["GET", "POST"])
def dashboard():
    positive, negative, plot_path = [], [], None
    sentiment_result, error = None, None
    tweet_text, keyword, bearer_token = "", "", ""

    if request.method == "POST":
        tweet_text = request.form.get("tweet_text", "")
        keyword = request.form.get("keyword", "")
        bearer_token = request.form.get("bearer_token", "")

        if tweet_text:
            try:
                sentiment_result = sentiment_pipeline(tweet_text[:512])[0]
            except Exception as e:
                error = f"Error analyzing tweet: {str(e)}"

        if bearer_token and keyword:
            positive, negative, plot_path, fetch_error = fetch_and_classify_tweets(bearer_token, keyword)
            if fetch_error:
                error = fetch_error

    return render_template(
        "index.html",
        tweet_text=tweet_text,
        sentiment_result=sentiment_result,
        keyword=keyword,
        bearer_token=bearer_token,
        positive=positive,
        negative=negative,
        plot_path=plot_path,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
