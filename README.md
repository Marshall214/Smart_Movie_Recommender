Manuel Smart AI - Movie Recommender

This is a simple yet smart content-based movie recommender system built using Python, Streamlit, and Natural Language Processing (NLP) techniques. It uses metadata like genres, cast, director, keywords, and movie overview to suggest similar movies.


Features

- Search for a movie and get top 5 similar movie recommendations
- Content-based filtering using TF-IDF vectorization
- Interactive UI built with Streamlit
- Fast, lightweight, and easy to deploy


How It Works

The recommender system builds a textual profile for each movie based on:

- Overview
- Genres
- Keywords
- Top 3 cast members
- Director

It then uses *TF-IDF vectorization* to convert text into vectors, and computes **cosine similarity** between them to find the most similar movies.


Dataset Used

- [`tmdb_5000_movies.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [`tmdb_5000_credits.csv`](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

These datasets are sourced from TMDb and contain movie metadata including crew and cast info.
