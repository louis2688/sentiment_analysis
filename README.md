# Product Review Sentiment Analysis

## Overview
This app processes a small dataset of product reviews, performs sentiment analysis, and categorizes the reviews as positive, negative, or neutral.

## Approach
- **Text preprocessing:** Tokenization, stop words removal, and lemmatization.
- **Sentiment analysis:** Using VADER sentiment analyzer.
- **Output:** CSV file with an additional sentiment column and a summary of sentiment categories.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place your `reviews.csv` file in the `data/` directory.
4. Run the script: `python main.py`
5. The results will be saved in the `output/` directory.

## Results
The analysis classified reviews into three categories with the following distribution:
- Positive: X reviews
- Negative: Y reviews
- Neutral: Z reviews
