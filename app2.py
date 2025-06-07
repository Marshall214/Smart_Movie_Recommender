import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from textblob import TextBlob
from transformers import pipeline
import spacy

# Load the dataset (same code as earlier)
movies_df = pd.read_csv("C:\\Datasets\\IMDB\\tmdb_5000_credits.csv")
credits_df = pd.read_csv("C:\\Datasets\\IMDB\\tmdb_5000_movies.csv")

movies_df = movies_df.merge(credits_df, left_on='movie_id', right_on='id')

# Load pre-trained models for sentiment analysis and summarization
summarizer = pipeline("summarization")
nlp = spacy.load("en_core_web_sm")

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
    return sentiment

# NER function to extract actors
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

# Parse movie names
def parse_names(obj):
    try:
        items = ast.literal_eval(obj)
        return [item['name'] for item in items]
    except:
        return []

# Apply to relevant columns
movies_df['genres'] = movies_df['genres'].apply(parse_names)
movies_df['keywords'] = movies_df['keywords'].apply(parse_names)
movies_df['cast'] = movies_df['cast'].apply(lambda x: parse_names(x)[:3])  # Top 3 actors

# Get director name
def get_director(obj):
    try:
        crew_list = ast.literal_eval(obj)
        for member in crew_list:
            if member['job'] == 'Director':
                return member['name']
        return np.nan
    except:
        return np.nan

movies_df['director'] = movies_df['crew'].apply(get_director)

# Create tags for recommendation
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['tags'] = (
    movies_df['overview'] + ' ' +
    movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['director'].fillna('')
)

# Final cleaned dataframe
final_df = movies_df[['id', 'title_x', 'tags', 'overview']]
final_df.dropna(inplace=True)
final_df.drop_duplicates(inplace=True)
final_df.reset_index(drop=True)

# Apply sentiment analysis
movies_df['sentiment'] = movies_df['overview'].apply(get_sentiment)

# Apply summarization
def summarize_text(text):
    if len(text) > 200:  # Summarize only long texts
        return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return text  # Return original text if it's already short

movies_df['summarized_overview'] = movies_df['overview'].apply(summarize_text)

# Vectorize the tags using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectorized_tfidf = tfidf.fit_transform(final_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectorized_tfidf)

# Recommendation system function
def recommend(movie, sentiment_threshold=0.2, actor_filter=None):
    movie = movie.lower()
    if movie not in final_df['title_x'].str.lower().values:
        return 'Movie not found in database'
    
    # Get the movie index
    index = final_df[final_df['title_x'].str.lower() == movie].index[0]
    
    # Apply sentiment filter
    movie_sentiment = movies_df.iloc[index]['sentiment']
    if movie_sentiment < sentiment_threshold:
        return 'Movie sentiment too low for recommendations'

    # Filter by actor if needed
    if actor_filter:
        recommendations = movies_df[movies_df['cast'].apply(lambda x: actor_filter in x)]
    else:
        # If no actor filter, use similarity scores
        distances = list(enumerate(similarity[index]))
        sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        recommendations = [final_df.iloc[i[0]] for i in sorted_movies]
    
    return recommendations

# Streamlit UI
st.title("Manuel Smart AI - Movie Recommender")

# User input for movie selection
movie_name = st.text_input("Enter a movie title:", placeholder="Enter a movie title")

# User input for actor filtering
actor_name = st.text_input("Filter by actor (optional):", placeholder="Enter actor's name (optional)")

# User input for sentiment filter
sentiment_slider = st.slider("Filter by sentiment:", min_value=-1.0, max_value=1.0, value=0.2)

if movie_name:
    # Call the recommendation function
    recommendations = recommend(movie_name, sentiment_threshold=sentiment_slider, actor_filter=actor_name)
    
    if isinstance(recommendations, list):
        st.write(f"Top recommendations for '{movie_name}':")
        for i, movie in enumerate(recommendations):
            st.write(f"{i + 1}. {movie['title_x']} - Sentiment: {movie['sentiment']:.2f}")
            st.write(f"Overview: {movie['summarized_overview']}")
    else:
        st.write(recommendations)
