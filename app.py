import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the dataset (same code as earlier)
movies_df = pd.read_csv("C:\\Datasets\\IMDB\\tmdb_5000_credits.csv")
credits_df = pd.read_csv("C:\\Datasets\\IMDB\\tmdb_5000_movies.csv")

movies_df = movies_df.merge(credits_df, left_on = 'movie_id', right_on = 'id')

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
final_df = movies_df[['id', 'title_x', 'tags']]
final_df.dropna(inplace=True)
final_df.drop_duplicates(inplace=True)
final_df.reset_index(drop=True)

# Vectorize the tags using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectorized_tfidf = tfidf.fit_transform(final_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectorized_tfidf)

# let's build the recommendation sysytem

def recommend(movie):
    movie = movie.lower()
    if movie not in final_df['title_x'].str.lower().values:
        return 'Movie not found in database'
    index = final_df[final_df['title_x'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in sorted_movies:
        recommendations.append(final_df.iloc[i[0]].title_x)
    
    return recommendations

# Streamlit UI
st.title("Manuel Smart AI - Movie Recommender")

# User input for movie selection with placeholder text
movie_name = st.text_input("", placeholder="Enter a movie title")

if movie_name:
    # Call the recommendation function
    recommendations = recommend(movie_name)
    
    if isinstance(recommendations, list):
        st.write(f"Top recommendations for '{movie_name}':")
        for i, movie in enumerate(recommendations):
            st.write(f"{i + 1}. {movie}")
    else:
        st.write(recommendations)

