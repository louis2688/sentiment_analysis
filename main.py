import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import ngrams
from nltk.tokenize import TreebankWordTokenizer

# Ensure NLTK data is downloaded and configured correctly
nltk.data.path.append('/Users/louis/nltk_data')
print("Using NLTK version:", nltk.__version__)
print("NLTK data path:", nltk.data.path)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('data/reviews.csv')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    # Tokenize the text using TreebankWordTokenizer to avoid issues with 'punkt'
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the review_text column
df['processed_text'] = df['review_text'].apply(preprocess_text)

# Display the processed data
print("Processed Data Sample:")
print(df[['review_text', 'processed_text']].head())

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the sentiment analysis
df['sentiment'] = df['processed_text'].apply(get_sentiment)

# Display the sentiment results
print("Sentiment Analysis Sample:")
print(df[['review_text', 'processed_text', 'sentiment']].head())

# Count the number of reviews in each sentiment category
summary = df['sentiment'].value_counts()

print("Summary of Sentiment Analysis:")
print(summary)

# Define the output file path
output_file = 'output/review_sentiments.csv'

# Save the DataFrame with the sentiment column to a CSV file
df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

# (Optional) Extract common keywords/phrases for positive and negative reviews
def get_common_phrases(sentiment, n=2):
    texts = df[df['sentiment'] == sentiment]['processed_text']
    all_phrases = []
    for text in texts:
        tokens = word_tokenize(text)
        phrases = ngrams(tokens, n)
        all_phrases.extend(phrases)
    return Counter(all_phrases).most_common(10)

# Get common phrases for positive and negative sentiments
positive_phrases = get_common_phrases('positive')
negative_phrases = get_common_phrases('negative')

print("Common Positive Phrases:")
for phrase, count in positive_phrases:
    print(f"{' '.join(phrase)}: {count}")

print("\nCommon Negative Phrases:")
for phrase, count in negative_phrases:
    print(f"{' '.join(phrase)}: {count}")
