import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading datasets...")

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# -------------------------
# Convert string dictionary to list
# -------------------------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# -------------------------
# Extract Top 3 Cast
# -------------------------
def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# -------------------------
# Extract Director
# -------------------------
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# -------------------------
# Clean Overview
# -------------------------
movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces between words
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# -------------------------
# Create Tags Column
# -------------------------
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

print("Vectorizing text...")

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

print("Calculating similarity matrix...")
similarity = cosine_similarity(vectors)

# Save pickle files
pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

print("Model Built Successfully!")
