
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv("movies.csv")  # Replace with your filename
df = df[['title', 'genres']].dropna()

# Vectorize genres
cv = CountVectorizer()
genre_matrix = cv.fit_transform(df['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Index map
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommend movies
def get_similar_movies(title, n=10):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation App")
movie_name = st.selectbox("Select a movie:", df['title'].values)
if st.button("Recommend"):
    recommendations = get_similar_movies(movie_name)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
