import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import ngrams
from nltk.tokenize import TreebankWordTokenizer

nltk.data.path.append('/Users/louis/nltk_data')
print("Using NLTK version:", nltk.__version__)
print("NLTK data path:", nltk.data.path)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('data/reviews.csv')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['review_text'].apply(preprocess_text)

print("Processed Data Sample:")
print(df[['review_text', 'processed_text']].head())

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['processed_text'].apply(get_sentiment)

print("Sentiment Analysis Sample:")
print(df[['review_text', 'processed_text', 'sentiment']].head())

summary = df['sentiment'].value_counts()

print("Summary of Sentiment Analysis:")
print(summary)

output_file = 'output/review_sentiments.csv'

df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

def get_common_phrases(sentiment, n=2):
    texts = df[df['sentiment'] == sentiment]['processed_text']
    all_phrases = []
    for text in texts:
        tokens = word_tokenize(text)
        phrases = ngrams(tokens, n)
        all_phrases.extend(phrases)
    return Counter(all_phrases).most_common(10)

positive_phrases = get_common_phrases('positive')
negative_phrases = get_common_phrases('negative')

print("Common Positive Phrases:")
for phrase, count in positive_phrases:
    print(f"{' '.join(phrase)}: {count}")

print("\nCommon Negative Phrases:")
for phrase, count in negative_phrases:
    print(f"{' '.join(phrase)}: {count}")
